Choosing the right lattice-dsp algorithm
========================================

.. admonition:: Tutorial goal

   Map common DSP tasks to the package APIs, diagnostics, and validation scope.

.. note::

   New to the terminology? See the :doc:`lattice DSP concept map <../../algorithms/concept_map>` and the :doc:`causality/data-use guide <../../theory/causality_and_data_use>` for how online, offline, block, and MIMO examples should be read.

Context
-------

New users often know the problem they want to solve before they know the package
vocabulary.  This tutorial gives a compact decision table for stable IIR filtering,
adaptive identification, AR estimation, finite-Hankel reduction, Nehari/AAK-style
finite diagnostics, MIMO reduction, and matrix-lattice experiments.

Key idea and equations
----------------------

The organizing principle is to choose the coordinate system that matches the
constraint: reflection coefficients for scalar stability, AR recursions for
prediction, Hankel singular values for input-output memory, and state-space Markov
parameters for MIMO reduction.

How to read the result
----------------------

Use the printed table as a routing guide, then follow the linked examples for the selected algorithm family.

Run command
-----------

.. code-block:: bash

   python examples/algorithm_selection_demo.py

Run status
----------

Return code: ``0``

Captured stdout
---------------

.. code-block:: text

   lattice-dsp algorithm-selection map
   ========================================
   1. stable scalar IIR filtering
      use: LatticeIIR, reflection_to_denominator
      diagnostic: reflection bounds and pole radius
      scope: stable scalar recursive filtering
   2. stable IIR with a numerator
      use: LatticeLadderIIR, numerator_to_ladder
      diagnostic: sample-by-sample agreement with B(z)/A(z)
      scope: lattice denominator plus ladder numerator
   3. adaptive recursive system identification
      use: AdaptiveLatticeLadderNLMS, LatticeLadderRLS
      diagnostic: learning curve, final MSE, learned pole radius
      scope: controlled online identification examples
   4. AR spectral estimation
      use: burg_denominator, levinson_durbin_denominator
      diagnostic: prediction error and spectrum residuals
      scope: stationary AR modeling diagnostics
   5. finite SISO model reduction
      use: hankel_singular_values, finite_hankel_reduce_iir
      diagnostic: Hankel singular-value decay and response error
      scope: finite-Hankel/Ho-Kalman baseline
   6. finite Nehari/AAK-style diagnostics
      use: finite_aak_siso_certificate, finite_aak_reduce_iir
      diagnostic: tail error, rational error, pole radius
      scope: finite-section candidate workflow
   7. MIMO state-space reduction
      use: mimo_state_space_markov_response, finite_hankel_reduce_mimo
      diagnostic: block-Hankel singular values and state-space response error
      scope: finite block-Hankel baseline
   8. matrix-lattice all-pass experiments
      use: MatrixLatticeAllPass, contractive_matrix_from_raw
      diagnostic: frequency-response singular values
      scope: all-pass/paraunitary tutorial scaffold
   
   representative tiny smoke checks
   denominator: [1.0, 0.4125, -0.25]
   leading_hankel_sv: [1.139654, 0.089349, 0.0]
   levinson_denominator: [1.0, 0.4125, -0.25]

Generated data files
--------------------

* :download:`algorithm_selection_map.csv <_artifacts/algorithm_selection_demo/algorithm_selection_map.csv>`

Source code
-----------

.. literalinclude:: ../../../examples/algorithm_selection_demo.py
   :language: python
   :linenos:
