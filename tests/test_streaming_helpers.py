from __future__ import annotations

import numpy as np

from lattice_dsp import AdaptiveBlockProcessor, BlockProcessor, LatticeIIR


def test_block_processor_matches_one_shot():
    rng = np.random.default_rng(8)
    x = rng.normal(size=1024)
    reflection = [0.35, -0.12]
    taps = [0.2, 0.1, 0.5]
    expected = np.asarray(LatticeIIR(reflection, taps).process(x), dtype=float)

    processor = BlockProcessor(reflection, taps)
    actual = np.concatenate([processor.process(x[i : i + 128]) for i in range(0, len(x), 128)])
    assert np.allclose(actual, expected)


def test_adaptive_block_processor_returns_block_result():
    rng = np.random.default_rng(9)
    x = rng.normal(size=512)
    desired = x.copy()
    processor = AdaptiveBlockProcessor([0.0], [0.0, 0.0], kind="rls")
    result = processor.process_adapt(x, desired)
    assert result.output.shape == x.shape
    assert result.error.shape == x.shape
