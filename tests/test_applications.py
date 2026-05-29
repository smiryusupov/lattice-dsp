import numpy as np

import lattice_dsp


def test_mse_and_improvement_db():
    clean = np.zeros(4)
    noisy = np.array([1.0, -1.0, 1.0, -1.0])
    improved = noisy * 0.5
    before = lattice_dsp.mse(clean, noisy)
    after = lattice_dsp.mse(clean, improved)
    assert before == 1.0
    assert after == 0.25
    assert lattice_dsp.improvement_db(before, after) > 5.9


def test_hybrid_echo_canceller_runs_and_returns_shapes():
    rng = np.random.default_rng(7)
    reference = rng.normal(size=1600)
    target = lattice_dsp.LatticeIIR([0.3, -0.15], [0.35, -0.05, 0.65])
    echo = np.asarray(target.process(reference), dtype=float)
    near = 0.02 * rng.normal(size=reference.size)
    microphone = echo + near

    canceller = lattice_dsp.HybridEchoCanceller(
        [0.0, 0.0],
        [0.0, 0.0, 0.0],
        mu_taps=0.05,
        mu_reflection=0.001,
        reflection_update_period=8,
    )
    result = canceller.process(reference, microphone, clean_target=near)

    assert result.echo_estimate.shape == reference.shape
    assert result.residual.shape == reference.shape
    assert result.enhanced.shape == reference.shape
    assert result.reflection.shape == (2,)
    assert result.taps.shape == (3,)
    assert np.isfinite(result.metrics["linear_improvement_db"])


def test_hybrid_echo_canceller_uses_residual_processor():
    rng = np.random.default_rng(8)
    reference = rng.normal(size=256)
    microphone = 0.5 * reference

    def halve(residual, context):
        assert "echo_estimate" in context
        assert context["sample_rate"] == 16_000
        return residual * 0.5

    canceller = lattice_dsp.HybridEchoCanceller(
        [0.0],
        [0.0, 0.0],
        residual_processor=halve,
        sample_rate=16_000,
        freeze_reflection=True,
    )
    result = canceller.process(reference, microphone)
    np.testing.assert_allclose(result.enhanced, result.residual * 0.5)


def test_make_residual_processor_from_model_supports_predict_protocol():
    class Model:
        def predict(self, residual, context):
            return residual - context["bias"]

    processor = lattice_dsp.make_residual_processor_from_model(Model())
    out = processor(np.array([1.0, 2.0]), {"bias": 0.25})
    np.testing.assert_allclose(out, [0.75, 1.75])


def test_spectral_residual_suppressor_preserves_shape_and_is_finite():
    rng = np.random.default_rng(11)
    signal = rng.normal(size=1024)
    suppressor = lattice_dsp.SpectralResidualSuppressor(
        frame_size=128,
        hop_size=32,
        floor=0.1,
        smoothing=0.2,
    )
    out = suppressor(signal, {"sample_rate": 16_000})
    assert out.shape == signal.shape
    assert np.isfinite(out).all()
    assert np.mean(out**2) <= np.mean(signal**2) * 1.05


def test_residual_attenuator_wrapper():
    signal = np.array([1.0, -2.0, 3.0])
    np.testing.assert_allclose(
        lattice_dsp.residual_attenuator(signal, gain=0.25), [0.25, -0.5, 0.75]
    )


def test_echo_aware_spectral_suppressor_uses_reference_context():
    rng = np.random.default_rng(22)
    reference = rng.normal(size=2048)
    near = rng.normal(size=2048)
    residual = 0.25 * reference + near
    suppressor = lattice_dsp.SpectralResidualSuppressor(
        frame_size=128,
        hop_size=32,
        mode="echo_aware",
        floor=0.05,
        smoothing=0.3,
    )
    out = suppressor(residual, {"echo_estimate": reference})
    assert out.shape == residual.shape
    assert np.isfinite(out).all()
    # Echo-aware suppression should not collapse the whole signal like an
    # aggressive blind residual gate would.
    assert np.mean(out**2) > 0.25 * np.mean(residual**2)


def test_spectral_residual_suppressor_mode_validation():
    try:
        lattice_dsp.SpectralResidualSuppressor(mode="unsupported")
    except ValueError as exc:
        assert "mode" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_hybrid_echo_canceller_safety_falls_back_on_divergent_residual():
    reference = np.ones(16)
    microphone = np.ones(16) * 0.01

    def explosive(residual, context):
        return residual * 1e9

    canceller = lattice_dsp.HybridEchoCanceller(
        [0.0],
        [0.0, 0.0],
        residual_processor=explosive,
        freeze_reflection=True,
        safety_max_power_ratio=10.0,
    )
    result = canceller.process(reference, microphone)
    assert result.metrics["safety_triggered"] == 1.0
    assert result.metrics["safety_stage"] == 2.0
    # The linear residual is kept when only the residual processor is unsafe.
    np.testing.assert_allclose(result.enhanced, result.residual)


def test_echo_gate_reduces_spectral_suppression_when_echo_estimate_is_weak():
    rng = np.random.default_rng(33)
    residual = rng.normal(size=2048)
    weak_echo = 1e-4 * rng.normal(size=2048)
    ungated = lattice_dsp.SpectralResidualSuppressor(
        frame_size=128,
        hop_size=32,
        mode="echo_aware",
        floor=0.05,
        smoothing=0.2,
        echo_aware_strength=0.9,
        echo_gate=False,
    )
    gated = lattice_dsp.SpectralResidualSuppressor(
        frame_size=128,
        hop_size=32,
        mode="echo_aware",
        floor=0.05,
        smoothing=0.2,
        echo_aware_strength=0.9,
        echo_gate=True,
        echo_gate_threshold=0.05,
        echo_gate_transition=0.20,
    )
    out_ungated = ungated(residual, {"echo_estimate": weak_echo})
    out_gated = gated(residual, {"echo_estimate": weak_echo})
    assert gated.last_echo_gate_scale < 0.1
    assert np.mean((out_gated - residual) ** 2) < np.mean((out_ungated - residual) ** 2)
