import argparse

import numpy as np

import lattice_dsp
from benchmarks.echo_cancellation_benchmark import fir_nlms, run_benchmark


def test_generate_echo_problem_shapes_and_components():
    problem = lattice_dsp.generate_echo_problem(samples=512, seed=10, nonlinear_strength=0.05)
    assert problem.reference.shape == (512,)
    assert problem.microphone.shape == (512,)
    assert problem.clean_target.shape == (512,)
    assert problem.linear_echo.shape == (512,)
    assert problem.nonlinear_echo.shape == (512,)
    assert problem.denominator[0] == 1.0
    assert np.isfinite(problem.microphone).all()


def test_echo_metrics_positive_for_better_output():
    clean = np.zeros(128)
    microphone = np.ones(128)
    enhanced = 0.25 * np.ones(128)
    metrics = lattice_dsp.echo_metrics(microphone, enhanced, clean)
    assert metrics.erle_db > 10.0
    assert metrics.mse_improvement_db > 10.0
    assert metrics.output_mse < metrics.input_mse


def test_fir_nlms_shapes():
    rng = np.random.default_rng(1)
    x = rng.normal(size=256)
    d = 0.5 * x
    y, e, w = fir_nlms(x, d, order=8, mu=0.2)
    assert y.shape == x.shape
    assert e.shape == x.shape
    assert w.shape == (8,)
    assert np.isfinite(e).all()


def test_echo_cancellation_benchmark_small_run():
    args = argparse.Namespace(
        samples=1024,
        sample_rate=16_000,
        seed=123,
        repeats=1,
        nonlinearity="tanh",
        nonlinear_strength=0.04,
        near_end_power_ratio=0.0,
        noise_snr_db=40.0,
        no_double_talk=True,
        iir_order=4,
        fir_order=16,
        fir_mu=0.4,
        mu_taps=0.05,
        mu_reflection=0.001,
        epsilon=1e-8,
        reflection_update_period=8,
        no_scale_reflection_mu_by_period=False,
        residual_gain=0.7,
    )
    payload = run_benchmark(args)
    names = {case["name"] for case in payload["cases"]}
    assert "no_cancellation" in names
    assert "lattice_iir_only" in names
    assert "lattice_iir_plus_toy_residual_suppressor" in names
    assert "spectral_residual_suppressor_only" in names
    assert "lattice_iir_plus_spectral_residual_suppressor" in names
    assert payload["best_by_erle"]["name"] in names
