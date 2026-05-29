from __future__ import annotations

import numpy as np

import lattice_dsp as ld


def _make_filter(rng: np.random.Generator, channels: int, order: int) -> ld.MatrixLatticeAllPass:
    reflections = [
        ld.contractive_matrix_from_raw(
            0.22
            * (rng.normal(size=(channels, channels)) + 1j * rng.normal(size=(channels, channels)))
        )
        for _ in range(order)
    ]
    residue = ld.unitary_polar_factor(
        rng.normal(size=(channels, channels)) + 1j * rng.normal(size=(channels, channels))
    )
    return ld.MatrixLatticeAllPass(reflections, residue=residue)


def test_matrix_lattice_paraunitary_block_reconstruction() -> None:
    rng = np.random.default_rng(12)
    channels = 3
    n_samples = 256
    filt = _make_filter(rng, channels, order=2)
    x = rng.normal(size=(n_samples, channels)) + 1j * rng.normal(size=(n_samples, channels))

    omega = 2.0 * np.pi * np.arange(n_samples) / n_samples
    response = filt.frequency_response(omega)
    spectrum = np.fft.fft(x, axis=0)
    y_spectrum = np.einsum("fij,fj->fi", response, spectrum)
    x_hat_spectrum = np.einsum("fji,fj->fi", response.conj(), y_spectrum)
    x_hat = np.fft.ifft(x_hat_spectrum, axis=0)

    assert np.linalg.norm(x_hat - x) / np.linalg.norm(x) < 1e-10
    assert (
        abs(
            np.vdot(np.fft.ifft(y_spectrum, axis=0), np.fft.ifft(y_spectrum, axis=0)).real
            - np.vdot(x, x).real
        )
        / np.vdot(x, x).real
        < 1e-10
    )


def test_matrix_lattice_real_allpass_preserves_block_energy() -> None:
    rng = np.random.default_rng(13)
    channels = 4
    n_samples = 512
    reflections = [
        ld.contractive_matrix_from_raw(0.35 * rng.normal(size=(channels, channels)))
        for _ in range(3)
    ]
    residue = ld.unitary_polar_factor(rng.normal(size=(channels, channels)))
    filt = ld.MatrixLatticeAllPass(reflections, residue=residue)
    x = rng.normal(size=(n_samples, channels))

    omega = 2.0 * np.pi * np.arange(n_samples) / n_samples
    response = filt.frequency_response(omega)
    spectrum = np.fft.fft(x, axis=0)
    y = np.fft.ifft(np.einsum("fij,fj->fi", response, spectrum), axis=0).real

    assert abs(np.sum(y * y) - np.sum(x * x)) / np.sum(x * x) < 1e-10
