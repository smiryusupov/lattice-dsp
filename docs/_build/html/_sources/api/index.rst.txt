API reference
=============

The public API is available from the top-level ``lattice_dsp`` package.  The
most important groups are listed below.

.. toctree::
   :maxdepth: 2

   top_level
   matrix_lattice
   model_reduction
   nehari
   multichannel_ar
   tangential_schur
   streaming
   applications
   metrics
   synthetic
   tuning

API groups
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Group
     - Main symbols
   * - Scalar lattice/IIR
     - ``LatticeIIR``, ``LatticeLadderIIR``, ``reflection_to_denominator``, ``denominator_to_reflection``
   * - Adaptive filters
     - ``LatticeLadderNLMS``, ``AdaptiveLatticeLadderNLMS``, ``LatticeLadderRLS``, ``adaptive_process_batch``, ``rls_process_batch``
   * - AR tools
     - ``autocorrelation``, ``levinson_durbin_*``, ``burg_*``
   * - Matrix lattice
     - ``MatrixLatticeAllPass``, ``OnlineMatrixLatticeAllPass``, ``online_matrix_lattice_allpass_process``, ``contractive_matrix_from_raw``, ``matrix_lattice_frequency_response``
   * - Finite-Hankel model reduction
     - ``finite_hankel_reduce_iir``, ``finite_hankel_reduce_mimo``, ``hankel_singular_values``
   * - Finite Nehari / rational candidates
     - ``finite_nehari_rational_candidates``, ``select_finite_nehari_candidate``, ``fit_rational_tail``
   * - Multichannel AR / block Levinson
     - ``multichannel_autocorrelation``, ``block_levinson_durbin``, ``MIMOLatticePredictor``, ``causal_mimo_lattice_predict``, ``solve_block_yule_walker_direct``
   * - Tangential Schur / J-inner
     - ``RightTangentialSchurData``, ``right_tangential_pick_matrix``, ``constant_schur_solution``, ``TangentialPotapovFactor``, ``PotapovProduct``
   * - Streaming
     - ``BlockResult``, ``BlockProcessor``, ``AdaptiveBlockProcessor``
   * - Metrics and synthetic demos
     - ``erle_db``, ``echo_metrics``, ``generate_echo_problem``

Note on C++ extension symbols
-----------------------------

Many core symbols are pybind11 bindings from ``lattice_dsp._core``.  To build API
docs with signatures available, install the package first:

.. code-block:: bash

   python -m pip install -e .[docs]
   cd docs
   make html
