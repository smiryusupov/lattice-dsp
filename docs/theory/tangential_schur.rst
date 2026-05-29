Tangential Schur, Pick matrices, and J-inner factors
====================================================

This page describes the experimental tangential-Schur layer in ``lattice-dsp``.
It is deliberately finite-dimensional and definite.  The goal is to give the
matrix/MIMO lattice examples a transparent interpolation-theory baseline without
claiming a complete generalized indefinite Schur solver or the full manifold
parametrization algorithms used in the Hanzon--Olivi--Peeters/Marmorat line of
work.

Schur functions and the RKHS/Pick viewpoint
-------------------------------------------

A matrix-valued Schur function is an analytic transfer matrix
:math:`S(z)\in\mathbb{C}^{p\times q}` on the unit disk with

.. math::

   \lVert S\rVert_\infty \le 1.

Equivalently, the de Branges--Rovnyak kernel

.. math::

   K_S(z,w)
   =
   \frac{I_q-S(z)^H S(w)}{1-\overline z w}

is positive.  Right tangential interpolation prescribes only selected input
directions:

.. math::

   S(z_i)U_i = V_i,
   \qquad |z_i|<1.

Here :math:`U_i\in\mathbb{C}^{q\times r_i}` contains input directions and
:math:`V_i\in\mathbb{C}^{p\times r_i}` contains the desired output vectors.  A
rank-one condition uses a single vector :math:`u_i` and value :math:`v_i`.

The finite Pick matrix implemented by the package is exactly the Gram matrix of
this kernel tested on the tangential columns:

.. math::

   P_{ij}
   =
   U_i^H K_S(z_i,z_j) U_j
   =
   \frac{U_i^H U_j - V_i^H V_j}
        {1-\overline{z_i}z_j}.

The definite Schur-class feasibility condition is

.. math::

   P\succeq 0.

This is the RKHS reason that the Pick matrix appears: feasible interpolation
data must make the finite kernel Gram matrix positive semidefinite.  In the
implemented module this is used as a finite certificate and conditioning
diagnostic, not as a black-box synthesis of all rational solutions.

Constant Schur solution helper
------------------------------

The package includes ``constant_schur_solution`` for the special case where the
data are compatible with a constant matrix contraction.  It solves

.. math::

   S_0 [U_1\;\cdots\;U_n] = [V_1\;\cdots\;V_n]

by a pseudoinverse and then checks both interpolation residual and contractivity.
This is not a general Schur-synthesis routine.  It is a useful sanity check for
examples because the exact solution is known.

Lossless/all-pass functions and J-inner matrices
------------------------------------------------

For a square transfer matrix :math:`Q(z)\in\mathbb{C}^{p\times p}`, a stable
lossless or inner all-pass function satisfies

.. math::

   Q(e^{j\omega})^H Q(e^{j\omega}) = I_p

for almost every point on the unit circle.  In state-space/DSP language this is
an energy-preserving stable all-pass system.  The package has a causal runtime
for such forward matrix-lattice all-pass filters through
``OnlineMatrixLatticeAllPass``.

Tangential Schur theory often works with a larger signature matrix

.. math::

   J=
   \begin{bmatrix}
      I_p & 0 \\
      0 & -I_q
   \end{bmatrix}

and with :math:`J`-inner transfer matrices :math:`\Theta(z)` satisfying

.. math::

   \Theta(e^{j\omega})^H J \Theta(e^{j\omega}) = J.

These :math:`J`-inner matrices are the natural objects in linear-fractional
Schur transformations.  They are related to, but not identical with, the square
all-pass runtime filters in the package: the runtime filters act on physical
signals, while the :math:`J`-inner factors act on graph vectors
:math:`[v;u]` and parameterize interpolation transformations.

Potapov--Blaschke factors
-------------------------

A scalar Blaschke factor for a point :math:`a` in the unit disk is

.. math::

   b_a(z)=\frac{z-a}{1-\overline a z},
   \qquad |b_a(e^{j\omega})|=1.

A Potapov--Blaschke factor is the matrix/signature analogue.  For strict
rank-one data :math:`S(a)u=v`, define the graph vector

.. math::

   \xi=\begin{bmatrix}v\\u\end{bmatrix},
   \qquad
   \xi^H J\xi=\lVert v\rVert_2^2-\lVert u\rVert_2^2<0.

The package forms the :math:`J`-orthogonal projection

.. math::

   P_\xi = \xi(\xi^H J\xi)^{-1}\xi^H J

and the elementary factor

.. math::

   \Theta_a(z)=I+(b_a(z)-1)P_\xi.

The two identities checked in code are

.. math::

   \Theta_a(e^{j\omega})^H J\Theta_a(e^{j\omega}) = J,
   \qquad
   \Theta_a(a)\xi=0.

Products of such factors remain :math:`J`-inner on the unit circle.  In the
package, ``TangentialPotapovFactor`` and ``PotapovProduct`` are used as
inspectable diagnostics: they expose the same algebraic invariants that underlie
matrix lossless/lattice theory, without claiming full generalized Schur
synthesis.

Schur parameters
----------------

In the scalar lattice/AR part of the package, Schur parameters are the familiar
reflection/PARCOR coefficients.  Bounded scalar Schur parameters
:math:`|k_i|<1` give a stable all-pole lattice denominator.

In multivariable tangential-Schur theory, a chart is usually described by chosen
interpolation points and directions together with contractive interpolation
values.  Informally, the values :math:`v_i` in

.. math::

   Q_i(w_i)u_i = v_i,
   \qquad \lVert v_i\rVert < 1,

play the role of Schur parameters in that chart.  The Hanzon--Olivi--Peeters and
Marmorat literature uses such parameters to build overlapping local
parametrizations of fixed-degree stable all-pass/lossless systems and to connect
them with balanced state-space realizations.

The current package does **not** implement that full recursive chart algorithm.
It implements the finite Pick matrix, constant-solution checks, and elementary
rank-one :math:`J`-inner/Potapov factors that make the surrounding algebra
visible and testable.

Relation to Hanzon--Olivi--Peeters/Marmorat algorithms
------------------------------------------------------

The implemented utilities are in the same mathematical ecosystem as the
real/complex tangential Schur algorithms used for manifolds of multivariable
stable all-pass systems, but they are not the same algorithm.

What those works use, in broad terms, is a recursive tangential-Schur procedure:
chosen chart data define interpolation conditions; each step applies a
linear-fractional transformation associated with a degree-one :math:`J`-inner
factor; the resulting Schur parameters provide local coordinates on a manifold
of fixed-degree lossless/all-pass systems, often with associated balanced
state-space realizations and chart-selection strategies.

What ``lattice-dsp`` implements in this release is narrower:

* finite right-tangential data containers;
* the definite Pick/RKHS Gram matrix :math:`P\succeq0`;
* residual checks for candidate transfer matrices;
* constant contractive solutions when the data are compatible with one;
* elementary rank-one Potapov--Blaschke/:math:`J`-inner factors;
* dense-grid :math:`J`-inner residual diagnostics.

Not implemented here:

* recursive extraction of Schur parameters from an arbitrary matrix inner
  function;
* an atlas of overlapping charts for fixed McMillan-degree lossless systems;
* balanced canonical state-space forms generated by the tangential-Schur
  recursion;
* manifold optimization for matrix :math:`H_2` approximation;
* generalized indefinite or singular tangential interpolation.

So the precise statement is: this module provides a tested finite definite
Pick/Potapov baseline that is compatible with the language used in that
literature, but it should not be described as the full Marmorat--Hanzon--Olivi--
Peeters tangential-Schur manifold algorithm.

Verification philosophy
-----------------------

The tangential-Schur implementation is tested through identities that are known
mathematically, rather than through one visually pleasing example.  The core
verification cases are:

* scalar data reduce to the classical Nevanlinna--Pick matrix;
* one-node rank-one data reduce to the norm condition
  :math:`\lVert v\rVert_2\le\lVert u\rVert_2`;
* data generated by a known constant contraction :math:`S_0` have
  :math:`P\succeq0` and recover :math:`S_0` when the tangential directions span
  the input space;
* diagonal MIMO data with coordinate directions decompose into independent
  scalar Pick blocks;
* feasible nonconstant scalar Schur data, for example
  :math:`S(z)=c b_a(z)`, pass the Pick test but are rejected by the
  constant-solution helper;
* tangential-column changes :math:`U_i\mapsto U_iQ_i`,
  :math:`V_i\mapsto V_iQ_i` transform the Pick matrix by a block-unitary
  congruence and preserve its eigenvalues;
* Potapov projections satisfy :math:`P_\xi^2=P_\xi` and
  :math:`P_\xi^HJ=JP_\xi`;
* elementary Potapov factors and their products satisfy the boundary identity
  :math:`\Theta(e^{j\omega})^HJ\Theta(e^{j\omega})=J` on dense frequency grids.

These tests are documented in :doc:`../algorithms/tangential_schur_verification`.
They are meant to make the finite definite baseline credible while still keeping
clear that full generalized Schur synthesis is outside the release scope.

Causality and data use
----------------------

Tangential Schur/Pick routines operate on finite interpolation data.  They are
not online time-series predictors and do not consume samples one at a time.  They
are best understood as **offline certificates and construction diagnostics**.

This is different from ``MIMOLatticePredictor`` or ``OnlineMatrixLatticeAllPass``:
those objects are causal runtimes.  Tangential Schur utilities explain part of
the mathematical background for matrix-lattice/all-pass structures, while the
online classes implement sample-by-sample DSP behavior.

Scope boundary
--------------

Implemented in this release:

* right tangential data containers;
* definite tangential Pick matrices and PSD checks;
* residual checks for candidate transfer matrices;
* constant contractive solutions when the data are compatible with one;
* elementary rank-one Potapov/J-inner factors;
* J-inner residual checks for individual factors and products.

Not claimed in this release:

* full generalized indefinite Schur interpolation;
* automatic synthesis of all rational Schur solutions from arbitrary Pick data;
* boundary interpolation and singular cases;
* complete MIMO inner--outer factorization.

The module is therefore a careful baseline rather than a black-box matrix-valued
interpolation solver.

Related tutorials
-----------------

* :doc:`../examples/generated/tangential_schur_pick_jinner`
* :doc:`../examples/generated/diagonal_tangential_schur_equals_scalar`
* :doc:`../examples/generated/matrix_lattice_allpass`
* :doc:`../examples/generated/mimo_diagonal_equals_independent_siso`
