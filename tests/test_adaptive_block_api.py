import numpy as np
import pytest

from lattice_dsp import AdaptiveLatticeLadderNLMS, LatticeIIR, adaptive_process_batch


def make_adaptive() -> AdaptiveLatticeLadderNLMS:
    return AdaptiveLatticeLadderNLMS(
        [0.0, 0.0],
        [0.0, 0.0, 0.0],
        mu_taps=0.08,
        mu_reflection=0.002,
        gradient_mode="analytic",
    )


def test_process_adapt_numpy_matches_legacy_list_block_errors():
    rng = np.random.default_rng(2026)
    x = rng.normal(size=512).astype(np.float64)
    target = LatticeIIR([0.25, -0.15], [0.4, -0.1, 0.2])
    desired = np.asarray(target.process(x), dtype=np.float64)

    legacy = make_adaptive()
    legacy_errors = np.asarray(legacy.adapt_block(x.tolist(), desired.tolist()), dtype=np.float64)

    block = make_adaptive()
    y, errors = block.process_adapt(x, desired)

    np.testing.assert_allclose(errors, legacy_errors, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(errors, desired - y, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(block.reflection, legacy.reflection, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(block.taps, legacy.taps, atol=1e-12, rtol=1e-12)


def test_process_adapt_rejects_mismatched_shapes():
    adaptive = make_adaptive()
    with pytest.raises(ValueError, match="same length"):
        adaptive.process_adapt(np.zeros(8), np.zeros(7))


def test_adaptive_process_batch_matches_independent_single_channel_runs():
    rng = np.random.default_rng(777)
    x = rng.normal(size=(3, 300)).astype(np.float64)
    target = LatticeIIR([0.2, -0.1], [0.3, -0.2, 0.1])
    desired = np.vstack([np.asarray(target.process(row), dtype=np.float64) for row in x])

    y_batch, e_batch, final_reflection, final_taps = adaptive_process_batch(
        [0.0, 0.0],
        [0.0, 0.0, 0.0],
        x,
        desired,
        mu_taps=0.08,
        mu_reflection=0.002,
        gradient_mode="analytic",
        n_threads=1,
    )

    assert y_batch.shape == x.shape
    assert e_batch.shape == x.shape
    assert final_reflection.shape == (x.shape[0], 2)
    assert final_taps.shape == (x.shape[0], 3)

    for ch in range(x.shape[0]):
        adaptive = make_adaptive()
        y, e = adaptive.process_adapt(x[ch], desired[ch])
        np.testing.assert_allclose(y_batch[ch], y, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(e_batch[ch], e, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(
            final_reflection[ch], adaptive.reflection, atol=1e-12, rtol=1e-12
        )
        np.testing.assert_allclose(final_taps[ch], adaptive.taps, atol=1e-12, rtol=1e-12)


def test_adaptive_process_batch_rejects_bad_input_shape():
    with pytest.raises(ValueError, match="2-D"):
        adaptive_process_batch([0.0], [0.0, 0.0], np.zeros(32), np.zeros(32))
