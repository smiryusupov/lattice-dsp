Interpolation, Schur stability, and why lattice coordinates help
=================================================================

This page explains the motivation behind several design choices in
``lattice-dsp``.  The short version is:

   Stable IIR design and reduction are often hard not because the formulas are
   unknown, but because the direct coefficient coordinates are numerically poor.
   Lattice, Schur, reflection, and Hankel coordinates expose the contractive and
   finite-rank structure more directly.

The discussion is intentionally finite-dimensional and numerical.  It gives the
mathematical map behind the tutorials; it is not a claim that the package solves
all Nevanlinna--Pick, AAK, or matrix-valued interpolation problems.

Why interpolation problems become numerically hard
-------------------------------------------------------

A classical scalar Nevanlinna--Pick problem asks for an analytic bounded function
``f`` on the unit disk that matches prescribed values,

.. math::

   f(z_i) = w_i, \qquad |f(z)| \le 1 \quad (|z| < 1).

In the Schur-class normalization, feasibility is characterized by the Pick
matrix

.. math::

   P_{ij} = \frac{1 - w_i \overline{w_j}}{1 - z_i \overline{z_j}}.

The exact mathematical condition is elegant: a solution exists when ``P`` is
positive semidefinite.  Numerically, however, this matrix can be very difficult:

* interpolation nodes close to each other produce nearly dependent kernel
  columns;
* nodes close to the unit circle amplify denominator sensitivity;
* values ``w_i`` close to the unit circle make the problem nearly extremal;
* high-order data can produce very large condition numbers even when the
  mathematical problem is feasible;
* tiny perturbations can change a nearly singular feasibility certificate into
  an apparently indefinite one.

This is the first reason direct coefficient methods are fragile.  Polynomial,
state-space, or denominator-coefficient solves can hide the contractive
structure that the problem is really about.  A numerically small coefficient
perturbation can correspond to a large movement of poles, zeros, or interpolation
residuals.

Why IIR and lattice coordinates are natural
-------------------------------------------

IIR filters are a natural representation for rational stable systems: a modest
number of poles can represent long memory.  The difficulty is that the denominator
controls stability.  In direct form,

.. math::

   H(z) = \frac{B(z)}{A(z)},
   \qquad
   A(z) = 1 + a_1 z^{-1} + \cdots + a_p z^{-p},

stability requires all roots of ``A`` to lie strictly inside the unit disk.  The
stable set in the direct coefficient vector ``a`` is not a simple box.  Adaptive
updates, least-squares fits, or model-reduction post-processing can easily cross
the stability boundary.

Reflection/PARCOR coefficients give a different scalar coordinate system.  In
Schur or Levinson recursion, a stable denominator is built stage by stage from
coefficients

.. math::

   k_1, k_2, \ldots, k_p,
   \qquad |k_m| < 1.

For scalar all-pole denominators, this boundedness condition is the key practical
benefit: the hard root-location condition becomes a per-stage contractivity
condition.  Implementations can enforce a margin, for example

.. math::

   |k_m| \le 1 - \varepsilon,

by clipping or by optimizing an unconstrained parameter through ``tanh``.  The
result is not a guarantee that every adaptive objective is solved well, but it
keeps the denominator inside a stable parameterization while the objective is
being optimized.

Inner and outer factors
-----------------------

Hardy-space language also explains why lattice and all-pass examples appear next
to Hankel and reduction examples.

A scalar Hardy-space transfer function can often be viewed through an
inner--outer factorization,

.. math::

   G(z) = G_{\mathrm{inner}}(z)\,G_{\mathrm{outer}}(z).

Informally:

* the **inner** factor is energy-preserving on the unit circle.  In SISO rational
  examples this is closely related to all-pass/Blaschke behavior;
* the **outer** factor carries the minimum-phase amplitude information and is
  tied to spectral factorization and prediction-error filters;
* Schur parameters and lattice stages are a constructive way to expose
  contractive/all-pass structure one stage at a time.

For MIMO systems, inner/outer factorization becomes matrix-valued.  The same
words still identify the conceptual roles, but the numerical problem is much
harder: factors may be rectangular, singular directions interact, and scalar
root tests are replaced by matrix contractivity and state-space conditions.  This
is why the package treats MIMO lattice/all-pass material as specialized
scaffolding and diagnostics rather than as a complete matrix-valued interpolation
solver.


Advanced context: tangential Schur and Potapov--Blaschke products
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Matrix-valued lattice filters are related to tangential Schur recursions and
Potapov--Blaschke products.  A scalar Blaschke factor is the simplest unit-disk
inner function; a Potapov--Blaschke factor is its matrix/signature analogue and
is :math:`J`-inner on the unit circle.  These factors are the building blocks
used in linear-fractional Schur transformations and in lossless/all-pass system
parametrizations.

The public API exposes a finite, definite baseline: right tangential Pick
matrices, RKHS/Pick positivity checks, interpolation residual checks,
constant-solution sanity checks, and elementary J-inner Potapov factors.  It
still does not claim full generalized indefinite matrix Schur synthesis or the
complete recursive Hanzon--Olivi--Peeters/Marmorat manifold parametrization.
See :doc:`tangential_schur` and
:doc:`../examples/generated/tangential_schur_pick_jinner`.

Kronecker theorem and finite Hankel rank
----------------------------------------

The Hankel point of view gives a second motivation for the model-reduction tools.
For a sequence of Markov parameters ``g_0, g_1, ...``, the Hankel matrix

.. math::

   H =
   \begin{bmatrix}
   g_1 & g_2 & g_3 & \cdots \\
   g_2 & g_3 & g_4 & \cdots \\
   g_3 & g_4 & g_5 & \cdots \\
   \vdots & \vdots & \vdots & \ddots
   \end{bmatrix}

summarizes how past inputs influence future outputs.  A classical Kronecker-type
statement says, in practical DSP terms:

   A Hankel operator has finite rank exactly when the corresponding sequence is
   generated by a rational transfer function; the rank corresponds to the
   realization order or McMillan degree under the usual minimality assumptions.

This is the reason singular values of finite Hankel blocks are useful.  Exact
finite rank suggests an exact low-order rational model.  Fast decay suggests
that a low-order model may be a good numerical approximation.  In SISO, this can
lead back to an IIR denominator and reflection coordinates.  In MIMO, the same
idea appears as block-Hankel/ERA-style state-space reduction.

How this motivates the package workflow
---------------------------------------

The package documentation uses the following chain:

.. code-block:: text

   interpolation and bounded analytic functions
      -> Schur contractivity and reflection coefficients
      -> stable scalar IIR/lattice coordinates
      -> Hankel finite-rank and model-reduction diagnostics
      -> MIMO block-Hankel/state-space examples
      -> matrix-lattice/all-pass scaffolds

The package is most useful when a user has data that can already be represented
as arrays: impulse responses, Markov parameters, time series, or synthetic room
responses.  The numerical routines then expose stability, order, and reduction
information in coordinates that are less fragile than direct polynomial updates.

Scope boundary
--------------

The page motivates the implementation but does not expand the public claims.  In
this release:

* scalar reflection coefficients are used as a practical stability
  parameterization for SISO IIR examples;
* finite Hankel and Ho--Kalman-style routines are finite numerical diagnostics
  and reducers;
* finite Nehari/AAK helpers expose finite-section candidate checks and
  certificates;
* MIMO routines operate on finite Markov tensors and reduced state-space models;
* definite right-tangential Pick/J-inner utilities are included as finite
  diagnostics; generalized indefinite matrix Schur synthesis, exact
  infinite-dimensional Nehari optimization, and complete MIMO inner--outer
  synthesis are outside the package scope.

Related tutorials
-----------------

* :doc:`../examples/generated/stability_vs_direct_iir`
* :doc:`../examples/generated/reflection_coefficients_stability_demo`
* :doc:`../examples/generated/reachability_observability_hankel_demo`
* :doc:`../examples/generated/mimo_coupled_model_reduction`
