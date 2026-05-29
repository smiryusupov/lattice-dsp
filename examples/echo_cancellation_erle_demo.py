"""Small synthetic echo-cancellation metric demo.

This example is intentionally modest.  It shows how the stable adaptive
lattice/IIR stage behaves on a known synthetic echo problem and how ERLE is
computed.  It is not a production acoustic echo canceller.
"""

from __future__ import annotations

import numpy as np

from lattice_dsp import HybridEchoCanceller, echo_metrics, generate_echo_problem


def fir_nlms(
    reference: np.ndarray,
    desired: np.ndarray,
    *,
    order: int = 64,
    mu: float = 0.5,
    epsilon: float = 1e-8,
):
    """Tiny FIR/NLMS reference baseline for this example."""

    w = np.zeros(order, dtype=np.float64)
    xbuf = np.zeros(order, dtype=np.float64)
    y = np.zeros_like(reference, dtype=np.float64)
    e = np.zeros_like(reference, dtype=np.float64)
    for n, sample in enumerate(reference):
        xbuf[1:] = xbuf[:-1]
        xbuf[0] = sample
        y_n = float(np.dot(w, xbuf))
        e_n = float(desired[n] - y_n)
        w += (mu * e_n / (float(np.dot(xbuf, xbuf)) + epsilon)) * xbuf
        y[n] = y_n
        e[n] = e_n
    return y, e, w


def main() -> None:
    problem = generate_echo_problem(
        samples=16_000,
        sample_rate=16_000,
        seed=123,
        nonlinear_strength=0.0,
        near_end_power_ratio=0.0,
        noise_snr_db=60.0,
        double_talk=False,
    )

    no_cancel = echo_metrics(problem.microphone, problem.microphone, problem.clean_target)

    _, fir_residual, _ = fir_nlms(problem.reference, problem.microphone, order=64, mu=0.5)
    fir_metrics = echo_metrics(problem.microphone, fir_residual, problem.clean_target)

    canceller = HybridEchoCanceller(
        initial_reflection=[0.0] * 4,
        initial_taps=[0.0] * 5,
        mu_taps=0.05,
        mu_reflection=0.001,
        reflection_update_period=8,
    )
    result = canceller.process(
        problem.reference, problem.microphone, clean_target=problem.clean_target
    )
    lattice_metrics = echo_metrics(problem.microphone, result.residual, problem.clean_target)

    print("Synthetic echo metric demo")
    print("--------------------------")
    print(f"No cancellation ERLE: {no_cancel.erle_db:8.2f} dB")
    print(f"FIR/NLMS ERLE:        {fir_metrics.erle_db:8.2f} dB")
    print(f"Lattice/IIR ERLE:     {lattice_metrics.erle_db:8.2f} dB")
    print()
    print("ERLE = 10 log10(input echo/error power / output residual error power).")
    print("Positive ERLE means residual error power was reduced. It does not by")
    print("itself prove speech quality, especially during double-talk.")
    print()
    print("Final lattice reflection coefficients:")
    print(np.array2string(result.reflection, precision=4))


if __name__ == "__main__":
    main()
