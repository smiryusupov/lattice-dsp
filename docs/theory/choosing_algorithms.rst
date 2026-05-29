Choosing an algorithm
=====================

``lattice-dsp`` is organized around stable recursive DSP coordinates.  This page
maps common tasks to the public algorithms and explains the validation boundary
for each family.  It is the recommended entry point for readers who know their
application goal but not the package terminology.

The package's uncommon niche is the bridge between stable SISO IIR lattice
filters, finite model-reduction diagnostics, and MIMO block-Hankel/matrix-lattice
experiments.  The table below is therefore organized by task rather than by
mathematical topic.

Decision table
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 32 43

   * - Goal
     - Use
     - Why this is the right starting point
   * - Stable scalar IIR filtering
     - ``LatticeIIR`` and ``reflection_to_denominator``
     - Reflection/PARCOR coefficients give a direct scalar stability
       parameterization: ``|k_i| < 1``.
   * - Stable IIR with a general numerator
     - ``LatticeLadderIIR``, ``numerator_to_ladder``, and ``ladder_to_numerator``
     - The denominator remains lattice-parameterized while ladder taps realize
       the numerator.
   * - Adaptive recursive system identification
     - ``AdaptiveLatticeLadderNLMS`` or ``LatticeLadderRLS``
     - Numerator and denominator parameters can be updated online while the
       reflection coefficients remain bounded.
   * - AR spectral estimation
     - Burg and Levinson-Durbin helpers
     - AR models, prediction-error filters, and reflection coefficients share
       the same lattice/Szegő recursion background.
   * - Comparing AR spectra with nonparametric spectra
     - Periodogram, Capon, and AR diagnostic examples
     - These examples show when a low-order AR model is informative and when it
       is too restrictive.
   * - Finite SISO model reduction
     - ``hankel_singular_values`` and ``finite_hankel_reduce_iir``
     - Finite Hankel singular values identify input-output memory directions;
       the reducer provides a finite-Hankel/Ho--Kalman baseline.
   * - Finite Nehari/AAK-style diagnostics
     - ``finite_nehari_approximate_tail``, ``finite_aak_siso_certificate``, and
       candidate-selection helpers
     - These expose finite-section diagnostics related to Hankel singular
       structure, rational tails, and stable candidate selection.
   * - MIMO state-space reduction
     - ``mimo_state_space_markov_response`` and ``finite_hankel_reduce_mimo``
     - Markov parameters are assembled into a block-Hankel matrix and reduced to
       a lower-order state-space model.
   * - Matrix-lattice and paraunitary experiments
     - ``MatrixLatticeAllPass`` and matrix reflection helpers
     - Matrix contraction stages provide stable/all-pass and paraunitary
       tutorial scaffolds for multichannel systems.
   * - Echo or equalization demonstrations
     - Synthetic examples and metrics such as ERLE, MSE, and improvement dB
     - These are controlled DSP demonstrations, not production acoustic or
       wireless systems.

How to decide by data type
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 32 34 34

   * - Input available
     - Recommended path
     - First diagnostic
   * - Reflection coefficients or a known stable denominator
     - Scalar lattice or lattice-ladder filtering
     - Pole radius and round-trip reflection conversion
   * - Time series with approximately AR structure
     - Burg or Levinson-Durbin AR estimation
     - Prediction error and spectral residuals
   * - Input/output pairs from an unknown recursive system
     - Adaptive lattice-ladder identification
     - Learning curve, final MSE, and learned pole radius
   * - Impulse response or IIR coefficients to simplify
     - Finite-Hankel SISO reduction
     - Hankel singular-value decay and response error
   * - Markov matrices from a MIMO state-space model
     - Block-Hankel MIMO reduction
     - Reduced order, state-space response error, and stability
   * - Need a unitary or all-pass multichannel transform
     - Matrix-lattice all-pass or paraunitary examples
     - Singular values of the frequency response

Validation boundary
-------------------

The public validation language is deliberately specific:

* scalar lattice filtering is validated as a stable recursive implementation
  coordinate system;
* adaptive examples are validated on controlled synthetic identification and
  tracking problems;
* finite-Hankel reduction is validated as a finite-dimensional Ho--Kalman-style
  approximation workflow;
* finite Nehari/AAK helpers are finite-section diagnostics and candidate
  workflows, not exact infinite-dimensional solvers;
* matrix/MIMO examples are diagnostic and tutorial scaffolds around block-Hankel,
  state-space, all-pass, and paraunitary constructions.

Out of scope for this release
-----------------------------

The package does not claim to provide:

* a general replacement for SciPy signal processing;
* a production acoustic echo canceller;
* a complete wireless-communications simulator;
* an exact infinite-dimensional SISO AAK/Nehari solver;
* a matrix-valued AAK/Nehari solver;
* a constructive dynamic matrix-lattice realization for arbitrary nonunitary MIMO
  state-space models.

Minimal workflow examples
-------------------------

Scalar stable IIR:

.. code-block:: python

   import numpy as np
   import lattice_dsp as ld

   reflection = np.array([0.7, -0.4])
   denominator = ld.reflection_to_denominator(reflection)
   y = ld.LatticeIIR(reflection, [1.0]).process(np.ones(128))

Finite-Hankel order selection:

.. code-block:: python

   import lattice_dsp as ld

   impulse = ld.iir_impulse_response(denominator, [1.0], 256)
   hsv = ld.hankel_singular_values(impulse, rows=48, cols=48)
   # Choose a reduced order after inspecting the decay in ``hsv``.

MIMO reduction from Markov parameters:

.. code-block:: python

   markov = ld.mimo_state_space_markov_response(A, B, C, D, n_samples=120)
   reduced = ld.finite_hankel_reduce_mimo(markov, reduced_order=4, block_rows=16, block_cols=16)

Related tutorials
-----------------

* :doc:`../examples/generated/algorithm_selection_demo`
* :doc:`../examples/generated/reflection_coefficients_stability_demo`
* :doc:`../examples/generated/reachability_observability_hankel_demo`
* :doc:`../examples/generated/finite_hankel_model_reduction`
* :doc:`../examples/generated/mimo_finite_hankel_model_reduction`
