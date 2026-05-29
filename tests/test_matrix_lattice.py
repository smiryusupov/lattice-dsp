import numpy as np

from lattice_dsp import (
    MatrixLatticeAllPass,
    contractive_matrix_from_raw,
    is_matrix_reflection_stable,
    matrix_lattice_stage_blocks,
    project_matrix_reflection,
    unitary_polar_factor,
)


def test_contractive_projection_and_stage_unitarity():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    k = contractive_matrix_from_raw(raw, margin=1e-4)
    assert is_matrix_reflection_stable(k, margin=1e-4)
    blocks = matrix_lattice_stage_blocks(k)
    t = np.block([[blocks[0], blocks[1]], [blocks[2], blocks[3]]])
    assert np.linalg.norm(t.conj().T @ t - np.eye(6)) < 1e-10


def test_matrix_lattice_allpass_response_is_unitary():
    rng = np.random.default_rng(1)
    ks = [
        contractive_matrix_from_raw(0.3 * (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))))
        for _ in range(3)
    ]
    residue = unitary_polar_factor(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    filt = MatrixLatticeAllPass(ks, residue=residue)
    omega = np.linspace(0, np.pi, 64)
    response = filt.frequency_response(omega, n_threads=1)
    assert response.shape == (64, 2, 2)
    assert filt.unitarity_error(omega) < 1e-9


def test_project_matrix_reflection_clips_spectral_norm():
    k = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=np.complex128)
    clipped = project_matrix_reflection(k, margin=1e-3)
    assert is_matrix_reflection_stable(clipped, margin=1e-3)
