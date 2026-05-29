"""Recipe: use Pyroomacoustics-style MIMO room impulse responses.

This example is dependency-free: it does not import Pyroomacoustics.  Instead it
uses a small fake object with the same ``room.rir[mic][source]`` structure used
by Pyroomacoustics after room impulse responses are computed.

In a real Pyroomacoustics script, build the room and compute its RIRs there, then
pass the resulting room object to ``markov_from_pyroom_rir`` below.
"""

from __future__ import annotations

import numpy as np

import lattice_dsp as ld


def markov_from_pyroom_rir(room, n_markov: int | None = None, dtype=float) -> np.ndarray:
    """Convert ``room.rir[mic][source]`` impulse responses to a MIMO Markov tensor.

    Parameters
    ----------
    room:
        Any object with a Pyroomacoustics-style ``rir`` attribute.
    n_markov:
        Optional number of taps to keep.  If omitted, the longest RIR length is
        used and shorter channels are zero-padded.
    dtype:
        Output dtype.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(n_taps, n_mics, n_sources)``.
    """

    n_outputs = len(room.rir)
    if n_outputs == 0:
        raise ValueError("room.rir must contain at least one microphone")
    n_inputs = len(room.rir[0])
    if n_inputs == 0:
        raise ValueError("room.rir[0] must contain at least one source")

    lengths = [len(room.rir[y][u]) for y in range(n_outputs) for u in range(n_inputs)]
    n_taps = max(lengths) if n_markov is None else int(n_markov)
    if n_taps <= 0:
        raise ValueError("n_markov must be positive")

    markov = np.zeros((n_taps, n_outputs, n_inputs), dtype=dtype)
    for y in range(n_outputs):
        if len(room.rir[y]) != n_inputs:
            raise ValueError("all microphones must have the same number of source RIRs")
        for u in range(n_inputs):
            h = np.asarray(room.rir[y][u], dtype=dtype)
            n = min(n_taps, h.size)
            markov[:n, y, u] = h[:n]
    return markov


class FakePyroomRoom:
    """Small stand-in for the Pyroomacoustics ``room.rir`` data structure."""

    def __init__(self, rir):
        self.rir = rir


def damped_room_path(delay: int, gain: float, decay: float, n_taps: int) -> np.ndarray:
    """Create a simple delayed, damped synthetic room path for the recipe."""

    h = np.zeros(n_taps)
    tail = gain * decay ** np.arange(n_taps - delay)
    h[delay:] = tail
    return h


def make_fake_room(n_taps: int = 96) -> FakePyroomRoom:
    """Create a 2-microphone, 2-source nested RIR list."""

    rir = [
        [
            damped_room_path(delay=3, gain=0.95, decay=0.91, n_taps=n_taps),
            damped_room_path(delay=8, gain=0.35, decay=0.88, n_taps=n_taps),
        ],
        [
            damped_room_path(delay=6, gain=0.42, decay=0.89, n_taps=n_taps),
            damped_room_path(delay=2, gain=0.82, decay=0.92, n_taps=n_taps),
        ],
    ]
    return FakePyroomRoom(rir)


def main() -> None:
    room = make_fake_room()
    markov = markov_from_pyroom_rir(room, n_markov=96)

    result = ld.finite_hankel_reduce_mimo(
        markov,
        reduced_order=4,
        block_rows=16,
        block_cols=16,
    )
    approx = ld.mimo_state_space_markov_response(
        result["A"], result["B"], result["C"], result["D"], markov.shape[0]
    )

    relative_error = float(np.linalg.norm(markov - approx) / np.linalg.norm(markov))
    hsv = np.asarray(result["hankel_singular_values"])

    print("converted shape:", markov.shape)
    print("mapping: markov[tap, microphone, source]")
    print("reduced order:", result["order"])
    print("stable reduced model:", bool(result["stable"]))
    print("leading block-Hankel singular values:", np.round(hsv[:6], 6).tolist())
    print("relative Markov-response error:", f"{relative_error:.3e}")

    print("\nReal Pyroomacoustics usage sketch:")
    print("  # import pyroomacoustics as pra")
    print("  # room = pra.ShoeBox(...); room.add_source(...); room.add_microphone_array(...)")
    print("  # room.compute_rir()")
    print("  # markov = markov_from_pyroom_rir(room, n_markov=512)")


if __name__ == "__main__":
    main()
