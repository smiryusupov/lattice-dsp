Tangential Schur and J-inner diagnostics
========================================

This algorithm page states what the implemented tangential-Schur utilities do in
computational terms.  For the mathematical background, see
:doc:`../theory/tangential_schur`.

Problem represented by the API
------------------------------

``RightTangentialSchurData`` stores finite interpolation conditions

.. math::

   S(z_i)U_i = V_i,
   \qquad S(z)\in\mathbb{C}^{p\times q},
   \qquad \lVert S\rVert_\infty \le 1.

The API supports one or more tangential columns at each point.  A rank-one datum
uses ``u_i`` and ``v_i`` vectors; a higher-multiplicity datum uses matrices
``U_i`` and ``V_i``.

Pick certificate
----------------

``right_tangential_pick_matrix`` builds the block matrix

.. math::

   P_{ij}=
   \frac{U_i^H U_j - V_i^H V_j}
        {1-\overline{z_i}z_j}.

``is_tangential_schur_solvable`` checks the finite definite certificate
:math:`P\succeq 0` by Hermitian eigenvalues.  This is an offline data test; it
does not construct a streaming filter.

Residual checks
---------------

``tangential_interpolation_residual`` and ``max_tangential_residual`` evaluate a
candidate constant matrix, a pointwise array of matrices, or a callable transfer
function against the data:

.. math::

   R_i = S(z_i)U_i - V_i.

These helpers are useful in examples because they make the exact interpolation
claim explicit.

Constant-solution sanity path
-----------------------------

``constant_schur_solution`` solves the special case

.. math::

   S_0[U_1\;\cdots\;U_n] = [V_1\;\cdots\;V_n]

and verifies that the recovered constant matrix is contractive.  This is not a
general finite Schur synthesis algorithm.  It is a stable baseline for tests and
examples where the true solution is intentionally known.

J-inner factor diagnostics
--------------------------

For strict rank-one data, ``TangentialPotapovFactor`` forms

.. math::

   \Theta(z)=I+(b_a(z)-1)P_\xi,
   \qquad
   \xi=\begin{bmatrix}v\\u\end{bmatrix},

with

.. math::

   J=\begin{bmatrix}I_p&0\\0&-I_q\end{bmatrix},
   \qquad
   \xi^H J\xi < 0.

The implemented checks are:

.. math::

   \Theta(e^{j\omega})^H J\Theta(e^{j\omega}) \approx J,
   \qquad
   \Theta(a)\xi \approx 0.

``PotapovProduct`` multiplies elementary factors that share the same signature
and checks the same boundary J-inner residual.

Potapov--Blaschke factors versus lossless filters
-------------------------------------------------

The scalar Blaschke factor

.. math::

   b_a(z)=\frac{z-a}{1-\overline a z}

has unit modulus on the unit circle.  The elementary Potapov factor replaces the
scalar ``1`` direction by a :math:`J`-orthogonal projection onto the graph vector
:math:`\xi=[v;u]`.  Therefore it is not just a plotting device: it is the
finite-dimensional object that makes the Schur step compatible with a
:math:`J`-inner linear-fractional transformation.

The package uses this in a diagnostic role.  ``OnlineMatrixLatticeAllPass`` is
the causal signal-processing runtime for square stable all-pass/lossless
filters.  ``TangentialPotapovFactor`` is the interpolation-side object used to
check the :math:`J`-inner algebra that underlies such lossless constructions.
The two are related through lossless/J-inner system theory, but the current
module does not automatically convert arbitrary tangential-Schur data into a
minimal causal matrix-lattice realization.

Schur parameters and scope versus manifold algorithms
-----------------------------------------------------

Scalar lattice filters use reflection/PARCOR coefficients as Schur parameters.
In tangential multivariable Schur algorithms, the chosen interpolation points,
directions, and contractive values form chart coordinates for lossless systems.
The Hanzon--Olivi--Peeters/Marmorat line uses recursive linear-fractional Schur
steps and balanced state-space realizations to parametrize manifolds of
fixed-degree stable all-pass systems.

This package does not implement that full recursive manifold algorithm.  It
implements the finite definite Pick certificate, constant-solution helper, and
rank-one Potapov/J-inner factors.  In other words, this is a tested baseline in
the same mathematical language, not a complete chart-selection or balanced-
realization solver.

RKHS interpretation of the Pick certificate
-------------------------------------------

For a Schur function :math:`S`, positivity of

.. math::

   K_S(z,w)=\frac{I-S(z)^HS(w)}{1-\overline z w}

is the reproducing-kernel Hilbert-space statement behind the Pick test.  The
implemented right-tangential Pick matrix is the finite Gram matrix

.. math::

   P_{ij}=U_i^H K_S(z_i,z_j)U_j.

Once :math:`V_i=S(z_i)U_i` is substituted, this gives the code formula

.. math::

   P_{ij}=\frac{U_i^HU_j-V_i^HV_j}{1-\overline z_i z_j}.

This RKHS view is useful for interpreting numerical eigenvalues: a small minimum
eigenvalue means the finite data are close to an extremal interpolation problem,
not necessarily that a time-domain filter is unstable.


Deep verification suite
-----------------------

The accompanying verification page :doc:`tangential_schur_verification` records
the mathematical invariants used by the test suite.  The most important checks
are scalar Pick reduction, random constant MIMO contractions, diagonal MIMO as a
direct sum of scalar Pick blocks, nonconstant scalar Schur data generated by a
Blaschke factor, unitary congruence under tangential-column basis changes,
near-boundary conditioning diagnostics, and J-inner residual checks for
individual Potapov factors and products.

This is intentionally stronger than a single example.  The package tests both
positive and negative cases: feasible data, infeasible data, constant-compatible
data, feasible-but-not-constant data, and invalid shapes/signatures.

Data-use status
---------------

These routines are **offline finite-data diagnostics**.  They are not causal
predictors and they do not consume a time stream.  Their role in the package is
to expose the Schur/Pick/J-inner structure that sits behind matrix-lattice and
all-pass examples.

Verification examples
---------------------

* :doc:`../examples/generated/tangential_schur_pick_jinner` checks Pick PSD,
  constant-solution residuals, graph-vector annihilation, and J-inner boundary
  residuals.
* :doc:`../examples/generated/diagonal_tangential_schur_equals_scalar` checks
  that diagonal MIMO tangential data reduce to independent scalar Pick problems.

Model-reduction stress-case connection
--------------------------------------

The tutorial :doc:`../examples/generated/mimo_model_reduction_stress_cases` uses
these routines as a finite sampled certificate on MIMO model-reduction data.  For
right-tangential samples ``F(z_i)u_i``, the response is scaled by ``gamma`` until
the Pick matrix is positive semidefinite.  This gives a Schur-feasibility
diagnostic for the sampled data and a tangential residual for reduced models.
The same page also plots H2/Markov error, finite-Hankel spectral-norm tail
error, and timing/backend information for the MIMO reducers.  It is
deliberately described as a diagnostic layer, not as a complete
tangential-Schur model-reduction algorithm.


Benchmark connection
--------------------

The benchmark :doc:`../benchmarks/generated/tangential_schur_mimo` measures the
same finite objects at small MIMO sizes: Pick-matrix construction,
PSD/eigenvalue checks, constant Schur recovery, Potapov product evaluation, and
J-inner residuals.  It also benchmarks the diagonal-MIMO reduction against
independent scalar Pick blocks, so the performance page mirrors the mathematical
sanity check used in the examples.
