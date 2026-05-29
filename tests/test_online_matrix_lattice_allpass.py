from __future__ import annotations

import numpy as np
import pytest

import lattice_dsp as ld


def _impulse_response(filt: ld.MatrixLatticeAllPass, n_samples: int) -> np.ndarray:
    h = np.empty((n_samples, filt.dimension, filt.dimension), dtype=np.complex128)
    for input_channel in range(filt.dimension):
        runtime = filt.to_online_filter()
        x = np.zeros((n_samples, filt.dimension), dtype=np.complex128)
        x[0, input_channel] = 1.0
        y = runtime.process(x)
        h[:, :, input_channel] = y
    return h


def test_online_matrix_lattice_scalar_order_one_matches_closed_form() -> None:
    k = 0.4
    filt = ld.MatrixLatticeAllPass([[[k]]], residue=[[1.0]])
    impulse = np.zeros((8, 1), dtype=np.complex128)
    impulse[0, 0] = 1.0
    y = filt.to_online_filter().process(impulse)[:, 0]

    s2 = 1.0 - k * k
    expected = np.empty(8, dtype=np.complex128)
    expected[0] = k
    for n in range(1, 8):
        expected[n] = s2 * ((-k) ** (n - 1))
    assert np.allclose(y, expected, atol=1e-14)


def test_online_matrix_lattice_impulse_response_matches_frequency_response() -> None:
    rng = np.random.default_rng(22)
    dim = 2
    reflections = [
        ld.contractive_matrix_from_raw(
            0.14 * (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim)))
        )
        for _ in range(2)
    ]
    residue = ld.unitary_polar_factor(
        rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    )
    filt = ld.MatrixLatticeAllPass(reflections, residue=residue)

    n_samples = 512
    h = _impulse_response(filt, n_samples)
    omega = np.linspace(0.0, np.pi, 33)
    powers = np.exp(-1j * np.outer(omega, np.arange(n_samples)))
    estimated = np.einsum("wn,nij->wij", powers, h)
    reference = filt.frequency_response(omega)
    rel = np.linalg.norm(estimated - reference) / np.linalg.norm(reference)
    assert rel < 2e-10


def test_online_matrix_lattice_process_is_prefix_causal() -> None:
    rng = np.random.default_rng(23)
    dim = 3
    reflections = [
        ld.contractive_matrix_from_raw(0.18 * rng.normal(size=(dim, dim))) for _ in range(2)
    ]
    residue = ld.unitary_polar_factor(rng.normal(size=(dim, dim)))
    filt = ld.MatrixLatticeAllPass(reflections, residue=residue)
    x = rng.normal(size=(64, dim))
    x_changed_future = x.copy()
    x_changed_future[25:] = rng.normal(size=x_changed_future[25:].shape)

    y = filt.to_online_filter().process(x)
    y_changed_future = filt.to_online_filter().process(x_changed_future)
    assert np.allclose(y[:25], y_changed_future[:25], atol=1e-13)


def test_online_matrix_lattice_diagonal_mimo_equals_independent_siso() -> None:
    channels = 3
    reflections = [np.diag([0.2, -0.15, 0.1]), np.diag([-0.08, 0.05, 0.03])]
    residue = np.diag([1.0, -1.0, 1.0])
    mimo = ld.MatrixLatticeAllPass(reflections, residue=residue)
    rng = np.random.default_rng(24)
    x = rng.normal(size=(256, channels))

    y_mimo = mimo.to_online_filter().process(x)
    y_siso = np.column_stack(
        [
            ld.MatrixLatticeAllPass(
                [np.array([[stage[ch, ch]]], dtype=np.complex128) for stage in reflections],
                residue=np.array([[residue[ch, ch]]], dtype=np.complex128),
            )
            .to_online_filter()
            .process(x[:, [ch]])[:, 0]
            for ch in range(channels)
        ]
    )
    assert np.allclose(y_mimo, y_siso, atol=1e-13)


def test_online_matrix_lattice_validation() -> None:
    filt = ld.MatrixLatticeAllPass([[[0.1]]])
    runtime = filt.to_online_filter()
    with pytest.raises(ValueError, match="sample must have shape"):
        runtime.process_sample([1.0, 2.0])
    with pytest.raises(ValueError, match="drain"):
        runtime.process([[1.0]], drain=-1)


def test_matrix_lattice_impulse_response_method_matches_local_helper() -> None:
    rng = np.random.default_rng(25)
    dim = 2
    reflections = [
        ld.contractive_matrix_from_raw(0.12 * rng.normal(size=(dim, dim))) for _ in range(2)
    ]
    filt = ld.MatrixLatticeAllPass(reflections)
    assert np.allclose(filt.impulse_response(32), _impulse_response(filt, 32), atol=1e-15)
    with pytest.raises(ValueError, match="n_samples"):
        filt.impulse_response(0)


def test_time_domain_convolution_matches_streaming_runtime() -> None:
    rng = np.random.default_rng(26)
    dim = 3
    reflections = [
        ld.contractive_matrix_from_raw(0.10 * rng.normal(size=(dim, dim))) for _ in range(3)
    ]
    residue = ld.unitary_polar_factor(rng.normal(size=(dim, dim)))
    filt = ld.MatrixLatticeAllPass(reflections, residue=residue)
    x = rng.normal(size=(80, dim)) + 1j * rng.normal(size=(80, dim))
    h = filt.impulse_response(160)

    y_stream = filt.to_online_filter().process(x, drain=20)
    y_conv = ld.matrix_lattice_impulse_response_convolution(x, h, drain=20)
    assert np.allclose(y_stream, y_conv, atol=2e-13)


def test_time_domain_finite_adjoint_satisfies_inner_product_identity() -> None:
    rng = np.random.default_rng(27)
    dim = 2
    reflections = [
        ld.contractive_matrix_from_raw(
            0.08 * (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim)))
        )
    ]
    filt = ld.MatrixLatticeAllPass(reflections)
    x = rng.normal(size=(40, dim)) + 1j * rng.normal(size=(40, dim))
    probe = rng.normal(size=(65, dim)) + 1j * rng.normal(size=(65, dim))
    h = filt.impulse_response(26)

    hx = ld.matrix_lattice_impulse_response_convolution(x, h, drain=25)
    h_adj_probe = ld.matrix_lattice_finite_adjoint(probe, h, output_length=x.shape[0])
    lhs = np.vdot(hx, probe)
    rhs = np.vdot(x, h_adj_probe)
    assert abs(lhs - rhs) < 1e-11


def test_time_domain_helpers_validate_shapes() -> None:
    h = np.ones((3, 2, 1), dtype=np.complex128)
    with pytest.raises(ValueError, match="x must have shape"):
        ld.matrix_lattice_impulse_response_convolution(np.ones((4, 2)), h)
    with pytest.raises(ValueError, match="y must have shape"):
        ld.matrix_lattice_finite_adjoint(np.ones((4, 1)), h)
    with pytest.raises(ValueError, match="output_length"):
        ld.matrix_lattice_finite_adjoint(np.ones((4, 2)), h, output_length=-1)
