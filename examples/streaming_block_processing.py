"""Stateful streaming/block processing helpers."""

from __future__ import annotations

import numpy as np

from lattice_dsp import AdaptiveBlockProcessor, BlockProcessor, LatticeIIR


def main() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=4096)
    reflection = [0.45, -0.2]
    taps = [0.3, 0.0, 0.55]

    one_shot = np.asarray(LatticeIIR(reflection, taps).process(x), dtype=float)
    stream = BlockProcessor(reflection, taps)
    pieces = [stream.process(x[i : i + 256]) for i in range(0, len(x), 256)]
    blockwise = np.concatenate(pieces)
    print("streaming matches one-shot:", np.allclose(one_shot, blockwise))

    desired = one_shot
    adaptive = AdaptiveBlockProcessor(reflection, [0.0, 0.0, 0.0], kind="rls")
    errors = []
    for i in range(0, len(x), 256):
        result = adaptive.process_adapt(x[i : i + 256], desired[i : i + 256])
        errors.append(result.error)
    err = np.concatenate(errors)
    print("RLS streaming tail MSE:", float(np.mean(err[-512:] ** 2)))
    print("RLS learned taps:", np.round(adaptive.adaptive.taps, 4))


if __name__ == "__main__":
    main()
