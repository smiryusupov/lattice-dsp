Matrix lattice API
==================

.. automodule:: lattice_dsp.matrix_lattice
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:


Causal online all-pass runtime
------------------------------

``MatrixLatticeAllPass`` defines the matrix-valued all-pass transfer function.
``OnlineMatrixLatticeAllPass`` is the corresponding causal time-domain runtime:

.. code-block:: python

   filt = lattice_dsp.MatrixLatticeAllPass(reflections, residue)
   runtime = filt.to_online_filter()
   y_t = runtime.process_sample(x_t)

The runtime output at time ``t`` uses the current input vector and previous
section states.  It does not use future input samples.  The API also exposes a
matrix impulse-response helper and a finite-record time-domain adjoint.  The
adjoint is useful for reconstruction diagnostics, but it is noncausal as a
streaming inverse because it depends on future output samples.

.. autosummary::

   lattice_dsp.MatrixLatticeAllPass
   lattice_dsp.OnlineMatrixLatticeAllPass
   lattice_dsp.online_matrix_lattice_allpass_process
   lattice_dsp.matrix_lattice_impulse_response_convolution
   lattice_dsp.matrix_lattice_finite_adjoint

Key top-level C++/pybind symbols
--------------------------------

The following matrix-lattice functions are also exported from ``lattice_dsp``:

.. autosummary::

   lattice_dsp.matrix_lattice_frequency_response
