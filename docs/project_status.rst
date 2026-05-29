Project scope and release validation
====================================

For the current public-alpha validation snapshot, including headline SISO and MIMO benchmark numbers, see :doc:`release_notes_0_1`.

Positioning summary
-------------------

The package is intentionally specialized.  Its public niche is the combination
of stable SISO IIR lattice/lattice-ladder filtering, finite model-reduction
diagnostics, and MIMO state-space/matrix-lattice experiments.  The MIMO layer is
highlighted because coupled MIMO block-Hankel reduction, compiled MIMO
state-space simulation, matrix-lattice all-pass bridge diagnostics, and finite
tangential Schur/Pick/J-inner utilities are a rare combination in a compact
Python DSP package.

This positioning is not a claim to be a complete MIMO solver stack.  The tables
below separate implemented finite-dimensional capabilities, diagnostic/tutorial
scaffolds, and out-of-scope solver claims.

This page separates supported APIs, tutorial/diagnostic workflows, dependency-free interoperability recipes, and capabilities that are outside the current public scope.

Implemented and intended for use
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Area
     - Current status
     - Main entry points
   * - Algorithm-selection and theory guides
     - Public-facing scope, validation boundaries, Pick interpolation conditioning, Schur/reflection stability coordinates, inner/outer factorization, Hardy/Hankel intuition, reachability, observability, and Kronecker finite-rank motivation.
     - :doc:`theory/choosing_algorithms`, :doc:`theory/hardy_hankel_state_space`, :doc:`theory/interpolation_schur_motivation`, :doc:`theory/tangential_schur`
   * - Dependency-free interoperability recipes
     - Public recipes for Pyroomacoustics-style room impulse responses, eSpeak/eSpeak NG WAV sources, and optional user-side audio loaders without required package dependencies.
     - :doc:`interoperability`, ``examples/pyroomacoustics_mimo_rir_recipe.py``
   * - Scalar lattice/IIR filtering
     - Stable scalar reflection/PARCOR and lattice-ladder filters backed by C++.
     - ``LatticeIIR``, ``LatticeLadderIIR``, ``process_batch``
   * - Adaptive lattice filters
     - NLMS/RLS-style adaptive tutorials and batch helpers for controlled examples.
     - ``AdaptiveLatticeLadderNLMS``, ``LatticeLadderRLS``
   * - AR and spectral diagnostics
     - Burg, Levinson-Durbin, periodogram, AR spectrum, and Capon tutorial diagnostics.
     - ``burg_reflection``, ``levinson_durbin_reflection``
   * - SISO finite-Hankel reduction
     - Finite-section Hankel SVD and Ho--Kalman-style reduced IIR models.
     - ``finite_hankel_reduce_iir``, ``hankel_singular_values``
   * - MIMO block-Hankel reduction, including coupled MIMO block-Hankel reduction
     - Finite block-Hankel/ERA-style state-space baseline from Markov matrices, plus compiled batched state-space simulation, a long-signal MIMO runtime stress tutorial, and three model-reduction stress cases covering slow 1/f tails, random rational 10-by-10 data, and an ill-conditioned high-degree 2-by-2 response, including order-70 and 400-state Hankel-tail diagnostics.
     - ``finite_hankel_reduce_mimo``, ``mimo_state_space_process_batch``
   * - Finite Nehari/rational candidates
     - Finite-dimensional teaching and validation workflow for anticausal tails and rational candidate selection.
     - ``finite_nehari_rational_candidates``
   * - Finite-section AAK/Nehari certificate
     - Schmidt-pair finite-section diagnostic with attached rational candidate.
     - ``finite_aak_siso_certificate``
   * - High-level finite-section SISO AAK/Nehari candidate
     - Candidate selection plus optional finite Schmidt-pair certificate for non-exact tails.
     - ``finite_aak_reduce_tail``
   * - Causal MIMO lattice prediction
     - Online vector lattice predictor from block-Levinson forward/backward reflection matrices; ``predict()`` runs before ``update(y_n)`` consumes the current vector.
     - ``MIMOLatticePredictor`` and ``examples/causal_mimo_lattice_prediction.py``
   * - Matrix/MIMO lattice tutorials
     - Working examples for causal matrix all-pass forward filtering, time-domain finite-record adjoint diagnostics, paraunitary-style analysis, multichannel AR, online MIMO lattice prediction, coupled-vs-SISO verification, and diagonal-MIMO sanity checks.
     - ``MatrixLatticeAllPass``, ``OnlineMatrixLatticeAllPass``, and examples
   * - Tangential Schur/Pick/J-inner baseline
     - Finite definite right-tangential data containers, Pick matrices, constant-solution checks, elementary Potapov factors, J-inner residual diagnostics, a MIMO scaling benchmark, and a documented verification map covering scalar reductions, diagonal-MIMO direct sums, random feasible/infeasible data, and J-inner invariants.
     - ``RightTangentialSchurData``, ``right_tangential_pick_matrix``, ``TangentialPotapovFactor``, :doc:`algorithms/tangential_schur_verification`, and :doc:`benchmarks/generated/tangential_schur_mimo`
   * - MIMO block-Hankel to matrix-lattice bridge diagnostics
     - Stable all-pass scaffold built from reduced Markov data; reports polar-factor gap as a diagnostic rather than an exact matrix-lattice realization.
     - ``examples/mimo_hankel_to_matrix_lattice_bridge.py``
   * - Experimental MIMO state-space to matrix-lattice realization scaffold
     - Solver-style all-pass/polar fit: searches Markov-initialized matrix-lattice candidates, optionally fits static gain compensation, and returns diagnostics.
     - ``experimental_mimo_state_space_to_matrix_lattice`` and ``fit_static_matrix_gains``

Diagnostic and tutorial scope
---------------------------------

These parts are included for learning, validation, and reproducible diagnostics.  They are not complete production solvers:

* algorithm-selection examples that route users to the right API family,
* Hardy/Hankel/state-space examples that explain reachability and observability diagnostics,
* finite Nehari/AAK tail approximation helpers,
* Schmidt-pair AAK diagnostics,
* finite-section AAK/Nehari certificates,
* finite rational candidate selection,
* high-level finite-section candidate reduction with ``finite_aak_reduce_tail``,
* synthetic echo/ERLE examples,
* interoperability recipes for bringing external RIR and WAV data into the array-based APIs without making Pyroomacoustics, eSpeak/eSpeak NG, librosa, or soundfile required dependencies,
* ML-inspired unitary convolution demos,
* MIMO model-reduction tutorials and benchmarks around the finite block-Hankel baseline, including 1/f, random rational, and ill-conditioned stress cases with H2/Hankel/timing plots, matrix-lattice bridge diagnostics, and an experimental all-pass/polar realization scaffold,
* experimental MIMO matrix-lattice realization sweeps that classify polar-fit, static-gain mismatch, and poor-scaffold regimes while staying separate from matrix AAK/Nehari optimality claims,
* tangential Schur examples that verify finite Pick certificates and J-inner factors without claiming generalized indefinite synthesis.

They are included so users can reproduce the finite-section diagnostics and understand the current scope boundaries.

Out of scope for this release
-----------------------------

The package does **not** claim:

* an exact infinite-dimensional Nehari solver,
* a production Adamjan--Arov--Krein rational approximation solver,
* a matrix-valued AAK/Nehari solver,
* an exact realization algorithm that converts arbitrary stable MIMO state-space gain responses into matrix-lattice all-pass form,
* a full generalized indefinite matrix Schur interpolation solver,
* a complete room-acoustics, WebRTC, or wireless-communications framework,
* a required compatibility layer for Pyroomacoustics, eSpeak/eSpeak NG, librosa, or soundfile.

Public model-reduction scope:

.. code-block:: text

   finite-Hankel / Ho--Kalman reduction: available now
   finite Nehari/rational candidate workflow: available now as a baseline
   finite-section AAK/Nehari certificate: available now as a diagnostic
   high-level finite-section SISO AAK/Nehari candidate reducer: available now
   full infinite-dimensional SISO AAK/Nehari reduction: outside current scope
   matrix-lattice scaffold from MIMO Markov data: available as a diagnostic bridge
   experimental matrix-lattice all-pass/polar fit with static gain diagnostics: available now as a scaffold
   exact dynamic matrix-lattice realization/fitting of arbitrary gain responses: outside current scope
   finite definite tangential Schur/Pick/J-inner diagnostics: available now
   generalized indefinite matrix Schur synthesis: outside current scope
   matrix AAK/Nehari reduction: outside current scope

Validation commands
-------------------

The all-in-one release validation wrapper is:

.. code-block:: bash

   ./scripts/release_validation.sh

Before a release or major commit, the equivalent manual commands are:

.. code-block:: bash

   python -m pip install -e '.[dev,examples,benchmark,docs]'
   PYTHONPATH="$PWD" pytest -q
   PYTHONPATH="$PWD" pytest -q tests/test_release_trust_claims.py
   ./scripts/build_docs_with_results.sh
   PYTHONPATH="$PWD" python tools/audit_public_api.py
   PYTHONPATH="$PWD" python tools/audit_public_api.py --hide-deprecated

For the model-reduction and Nehari path specifically:

.. code-block:: bash

   PYTHONPATH="$PWD" pytest -q \
     tests/test_hankel_aak_reduction.py \
     tests/test_finite_hankel_mimo_reduction.py \
     tests/test_finite_nehari_approximation.py \
     tests/test_finite_nehari_rational_api.py \
     tests/test_finite_nehari_rational_bridge.py \
     tests/test_finite_nehari_rank_sweep.py \
     tests/test_aak_siso_candidate_selection.py \
     tests/test_aak_siso_schmidt_pair_demo.py \
     tests/test_finite_aak_siso_certificate.py \
     tests/test_finite_aak_reduce_tail.py \
     tests/test_finite_aak_reduce_iir.py \
     tests/test_finite_aak_iir_reduction_speedup_benchmark.py \
     tests/test_coupled_mimo_lattice_filter.py \
     tests/test_mimo_hankel_to_matrix_lattice_bridge.py \
     tests/test_experimental_mimo_matrix_lattice_realization.py \
     tests/test_matrix_lattice_runtime_benchmark.py

Benchmark interpretation
------------------------

A representative finite-section IIR reduction benchmark with 32 channels and 30,000 samples/channel supports this interpretation:

* finite-Hankel reduction is the fastest practical baseline for many cases;
* finite-section AAK/Nehari candidates are useful for rank selection, Schmidt-pair
  certificates, rational diagnostics, and some high-quality reductions;
* neither finite-section path is a full infinite-dimensional AAK/Nehari solver.

The benchmark ``benchmarks/finite_aak_iir_reduction_speedup.py`` is part of release validation because it reports both favorable and unfavorable end-to-end cases.  When the finite-AAK candidate path is slower end-to-end for a case, the benchmark output documents that result directly.

A representative coupled MIMO benchmark with the compiled state-space backend,
3 inputs, 3 outputs, batch size 8, and 6,000 samples/batch showed strong
processing speedups after reduction, for example roughly 6x for full order 16
reduced to order 2--4 on one Linux/OpenMP machine.  The same table also showed
that one-shot end-to-end speedup can remain below one because block-Hankel
reduction is a preprocessing cost.  MIMO benchmark reports therefore quote processing speedup, one-shot end-to-end speedup, amortized end-to-end speedup, and break-even samples together.

The matrix-lattice runtime benchmark is a separate all-pass/scattering benchmark:
it compares the compiled matrix-lattice frequency-response evaluator with the
NumPy reference path and reports unitarity error.  The block-Hankel-to-lattice bridge tutorial is documented as a stable scaffold/initialization diagnostic, not an exact realization solver.

Generated artifacts
-------------------

Generated figures, JSON, CSV, and HTML are written to these locations:

* ``reports/`` for local manual runs,
* ``docs/examples/generated/_artifacts/`` for generated example artifacts,
* ``docs/benchmarks/generated/_artifacts/`` for generated benchmark artifacts,
* ``docs/_build/`` for rendered Sphinx HTML.

Release source trees contain scripts, tests, hand-written docs, and package code.  Local generated report output is reproducible from the documented commands.

Public release checklist
------------------------

Before tagging a public release, maintainers should verify the checks below.
They are included here so contributors can see the quality bar for the package.

#. the full test suite and release-trust claim tests pass locally and in CI;
#. Sphinx documentation builds without unexpected warnings;
#. examples and benchmarks write generated artifacts outside the repository root;
#. README and docs keep exact AAK/Nehari solvers, matrix AAK/Nehari solvers, and production MIMO solver stacks outside the current scope while documenting the compiled MIMO runtime separately from model reduction;
#. public APIs have docstrings and at least one tutorial or regression test;
#. ``tools/audit_public_api.py`` has been reviewed for accidental exports; and
#. compatibility aliases are documented when old names remain available.

