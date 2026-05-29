Recommended learning path
=========================

``lattice-dsp`` now contains several related pieces: stable scalar lattice
filters, adaptive IIR examples, finite-Hankel model reduction, finite-section
Nehari/AAK diagnostics, and matrix/MIMO lattice experiments.  This page gives a
suggested path through the documentation for new users.

Stage 0: orientation and theory map
-----------------------------------

Start here if the main question is: *what niche does this package cover, and
which algorithm should I use?*

#. :doc:`theory/package_positioning`

   Explains the uncommon combination of stable SISO IIR lattice filters,
   finite model-reduction diagnostics, MIMO block-Hankel reduction, and
   matrix-lattice all-pass scaffolds.

#. :doc:`theory/choosing_algorithms`

   Maps common DSP tasks to the package APIs, diagnostics, validation boundary,
   and out-of-scope areas.

#. :doc:`theory/filtering_relationships`

   Defines equalization, echo cancellation, prediction, adaptivity,
   recursivity, minimum phase, maximum phase, and the inner/outer viewpoint used
   by later lattice and model-reduction pages.

#. :doc:`examples/generated/algorithm_selection_demo`

   Prints the same decision map from executable code and runs tiny API smoke
   checks.

#. :doc:`theory/hardy_hankel_state_space`

   Introduces the Hardy-space, Hankel, reachability, and observability language
   used by the model-reduction pages.

Stage 0b: interoperability recipes
----------------------------------

Read this page early if you want the package-wide array shape conventions, or
if you already have state-space systems, room impulse responses, WAV files,
Schur/Pick data, or audio data from another package and want to use
``lattice-dsp`` without adding hard dependencies.

#. :doc:`interoperability`

   Shows dependency-free conventions for scalar streams, online MIMO streams,
   batched MIMO state-space data, MIMO Markov tensors, MIMO AR coefficients,
   matrix-lattice reflections, right tangential Schur/Pick data, SciPy and
   state-space bridges, MATLAB/Octave exchange, Pyroomacoustics-style
   ``room.rir`` data, eSpeak/eSpeak NG WAV generation, standard-library WAV
   loading, and optional librosa/soundfile user-side loaders.

#. :doc:`examples/generated/pyroomacoustics_mimo_rir_recipe`

   Converts ``room.rir[mic][source]`` into a MIMO Markov tensor and applies the
   finite block-Hankel reducer.

#. :doc:`examples/generated/external_audio_wav_recipe`

   Demonstrates the WAV-to-NumPy boundary without requiring an audio I/O
   dependency in the package.

Stage 1: scalar stability and lattice coordinates
-------------------------------------------------

Start here if the main question is: *why use lattice filters at all?*

#. :doc:`examples/generated/reflection_conversion`

   Reflection/PARCOR coefficients are the stable coordinate system behind the
   package.  This tutorial shows the round trip between reflection coefficients
   and denominator polynomials.

#. :doc:`examples/generated/reflection_coefficients_stability_demo`

   Shows reflection coefficients as a static stability coordinate system before
   the adaptive examples.

#. :doc:`examples/generated/stability_vs_direct_iir`

   Direct-form denominator updates can leave the stable region.  Bounded
   reflection updates keep the denominator in a stability-aware coordinate
   system.

#. :doc:`examples/generated/million_sample_iir_throughput`

   Shows the speed motivation for recursive models on million-sample signals:
   a compact stable IIR can represent a long acoustic-like decay with fixed
   state instead of a very long FIR tap vector.

#. :doc:`examples/generated/large_order_echo_stress`

   Extends the long-signal story to echo-scale model sizes: high-order stable
   recursive filtering is compared with the coefficient-traffic scale of a long
   FIR echo tap vector.

#. :doc:`algorithms/lattice_filters`

   Read this after the first two examples for the scalar IIR/lattice-ladder
   formulas.

Stage 2: adaptive recursive filtering
-------------------------------------

Move here once the scalar lattice coordinates make sense.

#. :doc:`examples/generated/adaptive_iir_system_identification`
#. :doc:`examples/generated/tracking_drifting_iir_system`
#. :doc:`examples/generated/hinf_lms_reproduction`
#. :doc:`algorithms/adaptive_filters`
#. :doc:`algorithms/robust_lms_hinf`

The H∞/minimax LMS tutorial is especially important for motivation: the same
simple adaptive recursion can be interpreted through a worst-case energy-gain
lens, not only as approximate least-squares optimization.

Stage 3: spectral diagnostics and AR models
-------------------------------------------

These examples give visual intuition for AR/lattice modeling before moving to
model reduction.

#. :doc:`examples/generated/periodogram_vs_ar_spectrum`
#. :doc:`examples/generated/capon_spectrum_demo`
#. :doc:`examples/generated/spectral_diagnostics_comparison`
#. :doc:`examples/generated/burg_levinson_ar_tools`
#. :doc:`algorithms/spectral_diagnostics`
#. :doc:`algorithms/ar_estimators`

Stage 4: finite-Hankel model reduction
--------------------------------------

This is the first model-reduction layer intended for practical use.

#. :doc:`examples/generated/reachability_observability_hankel_demo`

   Explains why finite Hankel singular values are tied to reachable and
   observable state directions.

#. :doc:`examples/generated/finite_hankel_model_reduction`
#. :doc:`benchmarks/generated/hankel_reduction_speedup`
#. :doc:`api/model_reduction`

The finite-Hankel reducer is a finite-section Ho--Kalman-style baseline.  It is
not advertised as exact AAK/Nehari reduction, but it is useful today and gives a
reference point for the finite-section APIs.

Stage 5: finite Nehari/AAK diagnostics and SISO IIR reduction
-------------------------------------------------------------

Read this sequence in order.  Each page adds one finite-dimensional layer.

#. :doc:`examples/generated/nehari_aak_siso_toy`
#. :doc:`benchmarks/generated/finite_nehari_rank_sweep`
#. :doc:`examples/generated/finite_nehari_rational_bridge`
#. :doc:`examples/generated/aak_siso_schmidt_pair_demo`
#. :doc:`examples/generated/aak_siso_candidate_selection`
#. :doc:`examples/generated/finite_nehari_exact_rational_tail`
#. :doc:`examples/generated/aak_siso_certificate_demo`
#. :doc:`examples/generated/finite_aak_noisy_tail_demo`
#. :doc:`examples/generated/finite_aak_iir_reduction_demo`
#. :doc:`benchmarks/generated/finite_aak_iir_reduction_speedup`

This sequence is the current SISO AAK/Nehari path.  It is best understood as a
finite-section workflow: rank selection, Schmidt-pair certification, rational
candidate fitting, exact-tail validation, and then application to stable IIR
filters.

Stage 6: matrix/MIMO intuition and baselines
--------------------------------------------

The matrix/MIMO material should be read after the scalar path.  The key bridge is
that a diagonal MIMO lattice is just independent SISO filters side by side.

#. :doc:`examples/generated/mimo_diagonal_equals_independent_siso`
#. :doc:`examples/generated/online_coupled_mimo_vs_siso`
#. :doc:`examples/generated/matrix_lattice_allpass`
#. :doc:`examples/generated/coupled_mimo_lattice_filter`
#. :doc:`examples/generated/paraunitary_filter_bank_demo`
#. :doc:`examples/generated/multichannel_levinson_ar`
#. :doc:`examples/generated/causal_mimo_lattice_prediction`
#. :doc:`examples/generated/mimo_finite_hankel_model_reduction`
#. :doc:`examples/generated/mimo_coupled_model_reduction`
#. :doc:`examples/generated/mimo_long_signal_state_space_stress`

   Stress the compiled MIMO runtime on long batched multichannel signals after
   finite block-Hankel reduction.  This is the clearest scale demonstration for
   the package's MIMO niche.

#. :doc:`examples/generated/mimo_hankel_to_matrix_lattice_bridge`
#. :doc:`examples/generated/experimental_mimo_matrix_lattice_realization`
#. :doc:`examples/generated/experimental_mimo_matrix_lattice_calibration`
#. :doc:`benchmarks/generated/matrix_lattice_runtime`
#. :doc:`algorithms/matrix_lattice`
#. :doc:`algorithms/multichannel_ar`

The MIMO block-Hankel reducer returns state-space matrices ``A, B, C, D`` from Markov parameters.  This is the reference MIMO baseline for the current release scope.  The bridge tutorial shows how reduced MIMO Markov data can seed a stable matrix-lattice all-pass scaffold.  The experimental realization
tutorial wraps this into a gain-search helper that fits the reduced model's
unitary polar factor.  The calibration tutorial then fits static left/right gains
around a known lattice response so readers can distinguish all-pass scaffold
error from static nonunitary gain mismatch.  The workflow remains an all-pass and
static-gain diagnostic scaffold rather than a full dynamic realization of
arbitrary nonunitary MIMO gain responses.

Status after this path
----------------------

After completing the path, the package scope is:

.. list-table::
   :header-rows: 1
   :widths: 35 45 20

   * - Layer
     - Role
     - Status
   * - Scalar lattice/IIR filtering
     - stable recursive filtering coordinates
     - implemented
   * - Finite-Hankel SISO reduction
     - practical finite-section model-reduction baseline
     - implemented
   * - Finite-section SISO AAK/Nehari workflow
     - rank selection, certificate, rational candidate, IIR reduction
     - implemented as finite-section baseline
   * - MIMO diagonal sanity check
     - shows diagonal MIMO equals independent SISO filters
     - implemented
   * - MIMO block-Hankel reduction
     - state-space baseline from Markov matrices
     - implemented
   * - Compiled MIMO state-space runtime
     - batched coupled MIMO state-space simulation for reduced/full model comparisons
     - implemented
   * - Matrix-lattice runtime and bridge diagnostics
     - all-pass runtime benchmark, coupled filter example, block-Hankel-to-lattice scaffold, and static gain calibration
     - implemented as diagnostics
   * - Full infinite-dimensional SISO AAK/Nehari solver
     - exact scalar solver beyond finite sections
     - outside current scope
   * - Matrix AAK/Nehari solver
     - MIMO/operator-theoretic solver
     - outside current scope
   * - Full MIMO state-space to matrix-lattice realization solver
     - constructive dynamic realization of general reduced MIMO models in matrix-lattice coordinates
     - outside current scope


MIMO reduction extension
------------------------

After the diagonal MIMO sanity check, read
:doc:`examples/generated/mimo_coupled_model_reduction` and then the coupled
MIMO benchmark.  This shows the finite block-Hankel baseline on a dense state-space system and keeps it separate from matrix AAK/Nehari solvers, which are outside the current scope.  The benchmark uses the compiled ``mimo_state_space_process_batch`` helper when available, which separates the runtime question from the reduction/certification question.


Theory motivation
-----------------

Readers who want the mathematical motivation behind the stable-IIR and
model-reduction examples should also read :doc:`theory/interpolation_schur_motivation`.
It explains why Pick matrices can be ill-conditioned, why Schur/reflection
coordinates are useful stability coordinates, how inner/outer factors relate to
all-pass and spectral-factor ideas, and why Kronecker finite-rank structure
motivates Hankel model reduction.
