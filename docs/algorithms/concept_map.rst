The lattice DSP concept map
===========================

How the classical ideas fit together
------------------------------------

``lattice-dsp`` uses terminology from several communities: speech linear
prediction, digital filters, orthogonal polynomials, operator theory,
rational approximation, multirate filter banks, and MIMO signal processing.
This page is a map of that vocabulary.  It is not meant to be a complete
mathematical treatment.  Its purpose is to help readers recognize why the
examples in the package belong together.  For the interpolation and numerical
conditioning motivation behind Schur/reflection coordinates, see
:doc:`../theory/interpolation_schur_motivation`.

The short version is:

.. pull-quote::

   Lattice filters are not just implementation tricks.  They are computational
   forms of Schur/Szegő recursions.  Reflection coefficients encode stability;
   prediction errors expose orthogonality; all-pass and paraunitary completions
   expose energy preservation; Hankel and Padé viewpoints explain rational
   approximation and model reduction; matrix/MIMO versions replace scalar
   reflection coefficients by contraction matrices.

1. The core object: a stable recursive filter
---------------------------------------------

The scalar IIR object is a rational transfer function

.. math::

   H(z) = \frac{B(z)}{A(z)}, \qquad
   A(z)=1+a_1 z^{-1}+\cdots+a_p z^{-p}.

For a causal discrete-time filter, stability is controlled by the roots of
``A``.  Directly updating the denominator coefficients ``a_i`` is fragile: a
small-looking coefficient update can move a pole outside the unit disk.  The
lattice viewpoint replaces the polynomial coefficients by reflection/PARCOR
coefficients

.. math::

   k_1, k_2, \ldots, k_p.

For the scalar all-pole recursion, the simple condition

.. math::

   |k_i| < 1 \quad \text{for all } i

is the practical stability handle used throughout the package.

Where to look: :doc:`lattice_filters`,
:doc:`Reflection conversion <../examples/generated/reflection_conversion>`, and
:doc:`Stability vs direct IIR <../examples/generated/stability_vs_direct_iir>`.

2. Schur recursion: stability by reflection coefficients
--------------------------------------------------------

The Schur recursion is the classical way to peel a stable analytic function or
stable polynomial into a sequence of reflection coefficients.  In signal
processing language, the same idea appears as step-up/step-down conversion
between denominator coefficients and PARCOR coefficients.

A typical step-up form is

.. math::

   A_m(z) = A_{m-1}(z) + k_m z^{-m} A_{m-1}^{\sharp}(z),

where ``sharp`` denotes the reversed/conjugate polynomial.  In real scalar code,
this becomes the familiar denominator update implemented by
``reflection_to_denominator``.

Practical interpretation:

* step-up recursion builds a stable denominator from reflection coefficients;
* step-down recursion tests or extracts reflection coefficients from a denominator;
* a reflection coefficient with magnitude at least one signals loss of stability.

Where to look: :doc:`Reflection conversion <../examples/generated/reflection_conversion>`.

3. Szegő polynomials: the orthogonal-polynomial view
----------------------------------------------------

Szegő recursions are the orthogonal-polynomial-on-the-unit-circle version of the
same structure.  Instead of viewing the recursion as a filter realization, one
views it as building orthogonal polynomials with respect to a spectral measure.
The reflection coefficients are then the unit-circle analogues of recursion
parameters; in the OPUC literature they are often called Verblunsky
coefficients.

For AR modeling, this is the conceptual bridge:

.. math::

   \text{autocorrelation moments}
   \quad \longrightarrow \quad
   \text{orthogonal prediction polynomials}
   \quad \longrightarrow \quad
   \text{reflection coefficients}.

That is why Levinson-Durbin, Burg estimation, prediction-error filters, and
lattice filters are so tightly connected.

Where to look: :doc:`ar_estimators`,
:doc:`Burg and Levinson AR tools <../examples/generated/burg_levinson_ar_tools>`, and
:doc:`AR spectral estimation <../examples/generated/ar_spectral_estimation>`.

4. Christoffel-Darboux: why prediction errors and kernels appear
----------------------------------------------------------------

The Christoffel-Darboux identity is a compact identity for sums of orthogonal
polynomials.  In this package it is not exposed as a public function, but it is
part of the background explaining why AR spectra, prediction-error energies,
and reproducing kernels are related.

A schematic version of the message is:

.. math::

   K_n(z,w) = \sum_{j=0}^n \phi_j(z)\overline{\phi_j(w)}

can be rewritten using only the latest orthogonal polynomials and their reversed
partners.  In filtering language, many cumulative prediction quantities can be
represented through the current forward/backward prediction-error polynomials
instead of through the whole history.

Practical interpretation:

* prediction-error filters are not arbitrary residual generators;
* they are orthogonalization objects;
* spectral peaks and prediction error are two views of the same AR model.

Where to look: :doc:`spectral_diagnostics`,
:doc:`Periodogram vs AR spectrum <../examples/generated/periodogram_vs_ar_spectrum>`, and
:doc:`Spectral diagnostics comparison <../examples/generated/spectral_diagnostics_comparison>`.

5. Orthogonal filters: lattice stages as rotations/reflections
--------------------------------------------------------------

A scalar lattice stage can be read as a local transformation of forward and
backward prediction errors.  In stable/lossless normalizations, lattice stages
behave like elementary rotations or hyperbolic rotations.  This is why lattice
filters appear naturally in speech coding, AR modeling, wave digital filters,
and filter-bank realizations.

For MIMO and matrix-valued systems, the scalar coefficient ``k`` is replaced by
a matrix ``K``.  The scalar bound becomes a contraction condition:

.. math::

   |k| < 1
   \qquad \leadsto \qquad
   \|K\|_2 < 1.

This is one of the most important analogies in the package.

Where to look: :doc:`matrix_lattice`,
:doc:`Matrix lattice all-pass <../examples/generated/matrix_lattice_allpass>`, and
:doc:`ML unitary convolution demo <../examples/generated/ml_unitary_convolution_demo>`.

6. All-pass completion: from a stable denominator to a lossless system
------------------------------------------------------------------------

An all-pass system has unit magnitude response in the scalar case:

.. math::

   |H(e^{i\omega})| = 1.

A matrix all-pass or paraunitary system satisfies the stronger matrix identity

.. math::

   H(e^{i\omega})^* H(e^{i\omega}) = I.

This means the system redistributes energy across time, frequency, or channels
without changing total energy.  Lattice and matrix-lattice structures are useful
because they can parameterize such energy-preserving systems through stable
local factors.

Practical interpretation:

* all-pass completions turn stability constraints into lossless systems;
* paraunitary filter banks are multichannel all-pass/orthogonal systems in
  polyphase form;
* MIMO lattice stages are compact parameterizations of structured unitary or
  contractive responses.

Where to look: :doc:`matrix_lattice`,
:doc:`Matrix unitary response compression <../examples/generated/matrix_unitary_response_compression>`,
:doc:`Paraunitary filter-bank demo <../examples/generated/paraunitary_filter_bank_demo>`.

7. Hankel forms and Padé: rational approximation and model reduction
--------------------------------------------------------------------

A recursive filter is a rational approximation to an input-output map.  Two
classical ways to understand reduced-order rational models are:

* **Padé approximation:** match a finite number of moments or Taylor/Markov
  coefficients;
* **Hankel/operator approximation:** approximate the map from past inputs to
  future outputs.

Given an impulse response

.. math::

   h_0, h_1, h_2, \ldots,

one finite Hankel matrix is

.. math::

   \mathcal H =
   \begin{bmatrix}
   h_1 & h_2 & h_3 & \cdots \\
   h_2 & h_3 & h_4 & \cdots \\
   h_3 & h_4 & h_5 & \cdots \\
   \vdots & \vdots & \vdots & \ddots
   \end{bmatrix}.

Its singular values indicate how much input-output memory is carried by
successive directions.  That is why Hankel singular values are natural order
selection diagnostics for reduced IIR models.

The current package includes finite model-reduction benchmarks and finite-section
Nehari/AAK-style diagnostics.  Exact infinite-dimensional Nehari/AAK reduction is
outside the public scope.  The theory pages explain this scope map.

Where to look: :doc:`../theory/hardy_hankel_state_space`,
:doc:`../theory/interpolation_schur_motivation`,
:doc:`model_reduction_hankel_aak`, and
:doc:`Model-reduction benchmark <../benchmarks/generated/model_reduction>`.

8. Matrix/MIMO generalization: contractions replace scalar coefficients
-----------------------------------------------------------------------

The matrix/MIMO path is the rare part of the package.  The scalar lattice story
has the following matrix analogue:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Scalar lattice idea
     - Matrix/MIMO analogue
   * - reflection coefficient ``k``
     - reflection/contraction matrix ``K``
   * - stability condition ``|k| < 1``
     - contraction condition ``||K||_2 < 1``
   * - scalar all-pass response
     - matrix all-pass/unitary response
   * - scalar AR prediction error
     - vector AR prediction-error covariance
   * - denominator polynomial
     - matrix polynomial / block Toeplitz system
   * - Levinson recursion
     - block Levinson / LWR-style recursion

This is why scalar lattice filters, multichannel AR, paraunitary filter banks,
and ML-inspired unitary convolutions are grouped together in the docs.  They are
all examples of structured stable or energy-preserving transformations.

Where to look: :doc:`multichannel_ar`, :doc:`matrix_lattice`,
:doc:`Multichannel Levinson AR <../examples/generated/multichannel_levinson_ar>`, and
:doc:`MIMO lattice vs block Levinson <../examples/generated/mimo_lattice_vs_block_levinson>`,
:doc:`Diagonal MIMO equals independent SISO <../examples/generated/mimo_diagonal_equals_independent_siso>`, and
:doc:`Finite block-Hankel MIMO reduction <../examples/generated/mimo_finite_hankel_model_reduction>`.

9. Where each concept appears in the package
--------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 44 32

   * - Concept
     - What it means
     - Where it appears
   * - Schur recursion
     - Converts stable denominators to and from reflection coefficients.
     - :doc:`Reflection conversion <../examples/generated/reflection_conversion>`;
       :doc:`Stability vs direct IIR <../examples/generated/stability_vs_direct_iir>`
   * - Szegő recursion
     - Orthogonal-polynomial version of the same reflection-coefficient recursion.
     - :doc:`ar_estimators`;
       :doc:`Burg/Levinson AR tools <../examples/generated/burg_levinson_ar_tools>`
   * - Schur/Jury stability test
     - Tests whether polynomial roots lie inside the unit disk; failed step-down
       coefficients reveal instability.
     - :doc:`lattice_filters`;
       :doc:`Stability vs direct IIR <../examples/generated/stability_vs_direct_iir>`
   * - Christoffel-Darboux identity
     - Reproducing-kernel identity behind compact prediction-error and spectral
       formulas.
     - :doc:`spectral_diagnostics`;
       :doc:`AR spectral estimation <../examples/generated/ar_spectral_estimation>`
   * - Orthogonal lattice filters
     - Lattice stages as local rotations/reflections of forward/backward errors.
     - :doc:`matrix_lattice`;
       :doc:`ML unitary convolution <../examples/generated/ml_unitary_convolution_demo>`
   * - All-pass completion
     - Completes a stable scalar or matrix factor into a lossless/energy-preserving
       response.
     - :doc:`Matrix lattice all-pass <../examples/generated/matrix_lattice_allpass>`;
       :doc:`Paraunitary filter bank <../examples/generated/paraunitary_filter_bank_demo>`
   * - Hankel forms
     - Encode input-output memory and moment/impulse-response matching.
     - :doc:`model_reduction_hankel_aak`;
       :doc:`Finite-Hankel SISO reduction <../examples/generated/finite_hankel_model_reduction>`;
       :doc:`Finite block-Hankel MIMO reduction <../examples/generated/mimo_finite_hankel_model_reduction>`
   * - Padé approximation
     - Builds rational approximants from moment or series matching.
     - :doc:`model_reduction_hankel_aak`; rational-approximation diagnostics
   * - Walsh and rational approximation viewpoints
     - Organize tables/families of rational approximants and explain degeneracies
       or near-cancellations.
     - Rational-approximation diagnostics
   * - Schmidt pairs / Schmidt form
     - Singular-vector structure of Hankel operators used in AAK theory.
     - :doc:`model_reduction_hankel_aak`; finite-section AAK diagnostics
   * - Matrix Schur / matrix lattice forms
     - MIMO generalization where reflection coefficients become contraction matrices.
     - :doc:`matrix_lattice`;
       :doc:`MIMO lattice vs block Levinson <../examples/generated/mimo_lattice_vs_block_levinson>`;
       :doc:`Diagonal MIMO equals independent SISO <../examples/generated/mimo_diagonal_equals_independent_siso>`

Suggested reading order
-----------------------

For a new reader, the most useful path is:

#. :doc:`lattice_filters` for scalar stability and reflection coefficients.
#. :doc:`ar_estimators` and :doc:`spectral_diagnostics` for prediction and spectra.
#. :doc:`adaptive_filters` and :doc:`robust_lms_hinf` for adaptive filtering.
#. :doc:`model_reduction_hankel_aak` for the model-reduction scope map.
#. :doc:`multichannel_ar` and :doc:`matrix_lattice` for the matrix/MIMO direction.

References
----------

For classical background, see :doc:`../references`, especially the sections on
scalar lattice filters, OPUC/Schur/Szegő theory, multirate/paraunitary systems,
and Hankel/Nehari/AAK model reduction.
