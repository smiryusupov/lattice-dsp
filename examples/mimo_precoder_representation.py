"""Toy MIMO-OFDM precoder representation with a matrix lattice all-pass filter.

This is deliberately a representation demo, not a complete 5G feedback system.
A low-order matrix lattice all-pass filter generates a unitary precoder across
many OFDM subcarriers using far fewer parameters than storing one dense matrix
per subcarrier.
"""

import numpy as np

from lattice_dsp import MatrixLatticeAllPass, contractive_matrix_from_raw, unitary_polar_factor

rng = np.random.default_rng(7)
dim = 4
order = 3
n_subcarriers = 1024

reflections = [
    contractive_matrix_from_raw(
        0.28 * (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim)))
    )
    for _ in range(order)
]
residue = unitary_polar_factor(rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim)))
precoder = MatrixLatticeAllPass(reflections, residue=residue)

omega = 2.0 * np.pi * np.arange(n_subcarriers) / n_subcarriers
v = precoder.frequency_response(omega)

stored_real_scalars = n_subcarriers * dim * dim * 2
lattice_real_scalars = precoder.parameter_count(real_scalars=True, include_residue=True)
print("MIMO dimension:", dim)
print("subcarriers:", n_subcarriers)
print("matrix lattice order:", order)
print("store all subcarrier matrices, real scalars:", stored_real_scalars)
print("matrix lattice representation, real scalars:", lattice_real_scalars)
print("compression ratio:", round(stored_real_scalars / lattice_real_scalars, 1), "x")
print("unitarity error:", f"{precoder.unitarity_error(omega[::32]):.3e}")
print("example precoder[0] first row:", np.round(v[0, 0], 3))
