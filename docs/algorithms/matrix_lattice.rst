Matrix lattice all-pass filters
===============================

For a compact checklist of the MIMO claims, examples, diagnostics, and data-use
status, see :doc:`mimo_verification_map`.

Motivation
----------

Scalar lattice filters parameterize stable IIR systems with reflection
coefficients satisfying ``|k_i| < 1``.  Matrix lattice filters generalize this
idea to multichannel transfer functions.  Each stage uses a matrix reflection
coefficient ``K_i``.  The scalar stability bound becomes a contractivity bound:

.. math::

   K_i^H K_i \prec I,

or equivalently, the largest singular value of ``K_i`` is strictly less than
one.

This lets the package build compact multichannel all-pass / paraunitary systems
whose frequency response is unitary:

.. math::

   H(e^{j\omega})^H H(e^{j\omega}) = I

for each frequency ``omega`` up to numerical precision.

Stage block construction
------------------------

A matrix-lattice stage can be written as a two-port unitary block.  Given a
contractive matrix ``K``, the package forms blocks involving positive-semidefinite
matrix square roots:

.. math::

   (I - K K^H)^{1/2},\quad (I - K^H K)^{1/2}.

The helper ``matrix_lattice_stage_blocks`` exposes these building blocks for
experimentation and debugging.

Stable parameter generation and projection
------------------------------------------

``contractive_matrix_from_raw`` maps an arbitrary raw complex matrix to a matrix
with spectral norm below a requested radius.  ``project_matrix_reflection`` clips
an existing matrix reflection to the stable/contractive set.  ``is_matrix_reflection_stable``
checks the condition.

Causality status
----------------

There are two separate MIMO notions in this package.

* ``mimo_state_space_process_batch`` is a causal state-space simulator.  It uses
  the recurrence

  .. math::

     q[n+1] = A q[n] + B u[n], \qquad
     y[n] = C q[n] + D u[n],

  so each output depends only on the current input and previous state.

* ``OnlineMatrixLatticeAllPass`` is a causal time-domain realization of a
  ``MatrixLatticeAllPass`` transfer function.  Each lattice section stores one
  delayed vector from the inner section.  With section blocks
  ``T11, T12, T21, T22``, one runtime step has the form

  .. math::

     y[n] = T_{11}u[n] + T_{12}d[n],\qquad
     v[n] = T_{21}u[n] + T_{22}d[n],

  followed by processing ``v[n]`` through the inner lattice and saving that
  inner output as the next delayed state ``d[n+1]``.  Therefore the output at
  time ``n`` uses only the current input and previous section states, never
  future samples.

The finite-record adjoint examples remain useful as block diagnostics,
especially when they check perfect reconstruction.  That adjoint is generally a
noncausal inverse of a stable causal all-pass, so streaming forward filtering
and finite-record synthesis are documented as separate runtime modes.  In this
release the examples use time-domain impulse-response adjoints rather than
FFT-domain circular multiplication when they need reconstruction diagnostics.

Frequency response and runtime
------------------------------

``MatrixLatticeAllPass.frequency_response(w)`` evaluates a batch of frequency
responses.  The C++ backend parallelizes over frequency points with OpenMP when
available.  ``MatrixLatticeAllPass.to_online_filter()`` creates an
``OnlineMatrixLatticeAllPass`` runtime for sample-by-sample causal processing,
and ``online_matrix_lattice_allpass_process`` is the matching convenience
wrapper.  The benchmark :doc:`../benchmarks/generated/matrix_lattice_runtime`
compares the compiled frequency-response path with the NumPy reference evaluator
and reports both speedup and unitarity error.

Streaming coupled filtering and finite-record adjoints
------------------------------------------------------

The tutorial :doc:`../examples/generated/coupled_mimo_lattice_filter` applies a
matrix-lattice all-pass to a complex multichannel signal with the causal online
runtime:

.. math::

   y[n] = \sum_{k\ge 0} H_k x[n-k].

The output at sample ``n`` depends only on the current input vector and previous
lattice states.  Because the response is all-pass, the full stream preserves
energy after the decaying tail is included.  For reconstruction diagnostics the
example applies the finite-record time-domain adjoint

.. math::

   x_{adj}[n] = \sum_{k\ge 0} H_k^H y[n+k].

This adjoint is noncausal as an online inverse because it requires future
transformed samples.  That distinction is intentional: the forward MIMO all-pass
is streaming, while perfect-reconstruction checks are finite-record diagnostics.

Diagonal MIMO sanity check
--------------------------

The easiest bridge from scalar to MIMO is the diagonal case.  If the Markov
matrices of a MIMO system are diagonal,

.. math::

   M_k = \operatorname{diag}(h_k^{(1)}, \ldots, h_k^{(p)}),

then the MIMO convolution separates into ``p`` independent SISO filters.  The
tutorial :doc:`../examples/generated/mimo_diagonal_equals_independent_siso`
uses five independent SISO lattice IIR filters to verify this equivalence.


Advanced context: tangential Schur and Potapov--Blaschke factors
----------------------------------------------------------------

Matrix-valued lattice filters are related to tangential Schur recursions and
Potapov--Blaschke products.  A square stable lossless/all-pass response satisfies

.. math::

   Q(e^{j\omega})^H Q(e^{j\omega})=I.

The causal ``OnlineMatrixLatticeAllPass`` runtime is the signal-processing side
of this story.  The tangential-Schur utilities are the interpolation side: they
work with graph vectors, Pick/RKHS positivity, and :math:`J`-inner Potapov--
Blaschke factors satisfying

.. math::

   \Theta(e^{j\omega})^H J\Theta(e^{j\omega})=J.

The public API exposes a finite, definite baseline: right tangential Pick
matrices, interpolation residual checks, constant-solution sanity checks, and
elementary J-inner Potapov factors.  It does not claim full generalized
indefinite matrix Schur synthesis or the complete Hanzon--Olivi--Peeters/
Marmorat recursive manifold parametrization.  See
:doc:`../theory/tangential_schur` and
:doc:`../examples/generated/tangential_schur_pick_jinner`.

Finite block-Hankel MIMO reduction
----------------------------------

For MIMO model reduction, the package uses Markov matrices rather than scalar
impulse samples.  ``finite_hankel_reduce_mimo`` builds a block-Hankel matrix
from those Markov parameters and returns a reduced state-space realization
``A, B, C, D``.  This is the MIMO analogue of the SISO finite-Hankel baseline,
not an exact matrix Nehari/AAK solver.

The bridge tutorial :doc:`../examples/generated/mimo_hankel_to_matrix_lattice_bridge`
then takes a reduced state-space model, extracts its frequency-response polar
factor, and compares it with a stable matrix-lattice all-pass scaffold initialized
from reduced Markov matrices.  The follow-up tutorial
:doc:`../examples/generated/experimental_mimo_matrix_lattice_realization` wraps
that idea in an experimental solver-style helper that searches over reflection
gains and returns the best all-pass/polar fit on a frequency grid.  The
calibration tutorial :doc:`../examples/generated/experimental_mimo_matrix_lattice_calibration`
then fits static left/right gains around a known lattice response.  This separates
all-pass scaffold error from static nonunitary gain mismatch.

This is still an all-pass/polar realization scaffold and possible initialization
strategy.  Static gain compensation is a diagnostic, not a dynamic realization
solver.  The workflow is **not** an exact algorithm for realizing arbitrary MIMO
state-space gain responses as matrix-lattice all-pass filters.

Why this is useful
------------------

Matrix lattice filters are not only a wireless precoder idea.  They are general
multichannel DSP primitives:

* streaming paraunitary analysis transforms with finite-record adjoint checks;
* streaming norm-preserving convolution blocks for ML experiments;
* multichannel audio decorrelation without energy loss;
* compact representations of frequency-dependent MIMO/unitary responses;
* stable all-pass scaffolds for reduced MIMO models;
* unitary filter-bank and invertible-transform prototypes.

Relevant APIs
-------------

* ``MatrixLatticeAllPass``
* ``OnlineMatrixLatticeAllPass``
* ``online_matrix_lattice_allpass_process``
* ``matrix_lattice_impulse_response_convolution``
* ``matrix_lattice_finite_adjoint``
* ``matrix_lattice_frequency_response``
* ``fit_static_matrix_gains``
* ``contractive_matrix_from_raw``
* ``project_matrix_reflection``
* ``is_matrix_reflection_stable``
* ``matrix_lattice_stage_blocks``
* ``matrix_spectral_norm``
* ``unitary_polar_factor``
* ``psd_matrix_sqrt``
* ``finite_hankel_reduce_mimo``
* ``experimental_mimo_state_space_to_matrix_lattice``
* ``mimo_state_space_frequency_response``
* ``polar_factor_response``
* ``matrix_lattice_scaffold_from_markov``
* ``mimo_state_space_markov_response``
* ``mimo_state_space_process_batch``

Examples
--------

* ``examples/mimo_diagonal_equals_independent_siso.py``
* ``examples/causal_mimo_lattice_prediction.py``
* ``examples/online_coupled_mimo_vs_siso.py``
* ``examples/mimo_finite_hankel_model_reduction.py``
* ``examples/mimo_coupled_model_reduction.py``
* ``examples/mimo_hankel_to_matrix_lattice_bridge.py``
* ``examples/matrix_lattice_allpass.py``
* ``examples/coupled_mimo_lattice_filter.py``
* ``examples/matrix_unitary_response_compression.py``
* ``examples/paraunitary_filter_bank_demo.py``
* ``examples/ml_unitary_convolution_demo.py``
* ``examples/multichannel_audio_decorrelator.py``

References
----------

Modern matrix-lattice precoder literature is a useful reference point because it
uses matrix all-pass lattice filters as compact unitary MIMO representations and
emphasizes stability and tracking through lattice parameters.  General
paraunitary and filter-bank background is covered by Vaidyanathan's multirate
text; orthogonal convolution work connects paraunitary systems to ML layers.
See :doc:`../references`.

The benchmark page :doc:`../benchmarks/generated/experimental_mimo_matrix_lattice_realization_sweep` sweeps reduced and lattice orders for the experimental scaffold.
