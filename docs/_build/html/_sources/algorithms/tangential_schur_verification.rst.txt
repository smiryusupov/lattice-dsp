Tangential Schur verification map
=================================

The tangential-Schur layer is intentionally conservative: it implements finite
right-tangential Pick certificates, constant-solution sanity paths, and
rank-one Potapov/J-inner diagnostics.  This page records the mathematical
invariants used by the examples and tests so the feature can be audited without
reading the implementation first.

Problem and certificate
-----------------------

For right tangential data

.. math::

   S(z_i)U_i = V_i,
   \qquad S : \mathbb{D}\to\mathbb{C}^{p\times q},
   \qquad \lVert S\rVert_\infty \le 1,

with :math:`U_i\in\mathbb{C}^{q\times r_i}` and
:math:`V_i\in\mathbb{C}^{p\times r_i}`, the definite Pick matrix is

.. math::

   P_{ij}
   =
   \frac{U_i^H U_j - V_i^H V_j}
        {1-\overline{z_i}z_j}.

The implemented finite feasibility certificate is

.. math::

   P \succeq 0.

This is the central invariant.  All tests either construct data where this
certificate is known to hold, construct data where it must fail, or verify that
algebraic transformations preserve it.

Verification cases
------------------

.. list-table:: Tangential-Schur verification checklist
   :header-rows: 1
   :widths: 26 38 36

   * - Claim
     - Mathematical check
     - Where it is tested or shown
   * - Scalar reduction equals classical Pick interpolation
     - With :math:`u_i=1`, :math:`v_i=w_i`, recover
       :math:`(1-\overline{w_i}w_j)/(1-\overline{z_i}z_j)`.
     - ``test_right_tangential_pick_matrix_scalar_reduces_to_classical_pick``
   * - One-node feasibility is the norm condition
     - For one rank-one datum,
       :math:`P_{11}\ge0 \Longleftrightarrow \lVert v\rVert\le\lVert u\rVert`.
     - ``test_single_node_norm_condition_is_exact_for_rank_one_data``
   * - Constant MIMO contractions are feasible
     - If :math:`V_i=S_0U_i` and :math:`\lVert S_0\rVert_2\le1`, then
       :math:`P\succeq0`; when the directions span the input space,
       ``constant_schur_solution`` recovers :math:`S_0`.
     - ``test_random_constant_mimo_data_are_recovered_across_shapes``
   * - Nonconstant Schur data need not be constant-compatible
     - Data from :math:`S(z)=c b_a(z)` are feasible for :math:`|c|<1`, but
       generally fail the constant-solution helper.
     - ``test_scalar_blaschke_data_are_feasible_but_not_constant``
   * - Tangential-column basis changes are congruences
     - Replacing :math:`U_i,V_i` by :math:`U_iQ_i,V_iQ_i` gives
       :math:`P' = Q^H P Q` with block-diagonal unitary :math:`Q`.
     - ``test_pick_matrix_is_hermitian_and_unitary_mixing_is_congruent``
   * - Diagonal MIMO decomposes into scalar Pick blocks
     - Coordinate directions and diagonal values make the MIMO Pick matrix a
       direct sum of scalar Pick matrices.
     - ``test_diagonal_mimo_tangential_data_decompose_into_scalar_pick_blocks``
       and :doc:`../examples/generated/diagonal_tangential_schur_equals_scalar`
   * - Near-boundary data remain explicit diagnostics
     - Points close to :math:`|z|=1` can make :math:`P` ill-conditioned; the
       eigenvalues should remain finite and visible rather than hidden.
     - ``test_near_boundary_data_remain_finite_and_report_conditioning``
   * - Graph-vector strictness is checked before Potapov factors
     - For :math:`\xi=[v;u]` and
       :math:`J=\operatorname{diag}(I_p,-I_q)`, require
       :math:`\xi^H J\xi<0`.
     - ``test_elementary_potapov_factor_rejects_non_strict_data``
   * - Elementary factors are J-inner on the boundary
     - :math:`\Theta(e^{j\omega})^HJ\Theta(e^{j\omega})=J` and
       :math:`\Theta(z_i)\xi_i=0`.
     - ``test_j_inner_factor_preserves_j_form_on_many_boundary_points`` and
       :doc:`../examples/generated/tangential_schur_pick_jinner`
   * - Products of compatible factors remain J-inner
     - Products share the same :math:`J` signature and preserve the boundary
       identity up to numerical tolerance.
     - ``test_potapov_product_is_j_inner_on_boundary`` and
       ``test_potapov_product_rejects_mixed_j_signatures``

J-inner arithmetic
------------------

For a strict rank-one datum, define

.. math::

   \xi = \begin{bmatrix} v \\ u \end{bmatrix},
   \qquad
   J = \begin{bmatrix} I_p & 0 \\ 0 & -I_q \end{bmatrix},
   \qquad
   \xi^H J\xi < 0.

The implemented elementary factor is

.. math::

   \Theta(z)=I+(b_a(z)-1)P_\xi,
   \qquad
   P_\xi = \xi(\xi^H J\xi)^{-1}\xi^HJ.

The projection is checked through

.. math::

   P_\xi^2=P_\xi,
   \qquad
   P_\xi^H J = JP_\xi,

and the boundary identity

.. math::

   \Theta(e^{j\omega})^HJ\Theta(e^{j\omega})=J.

Because the scalar Blaschke factor satisfies :math:`|b_a(e^{j\omega})|=1`, the
J-orthogonal projection formula gives a direct finite-dimensional diagnostic for
J-inner arithmetic.  The package tests this on individual factors, products, and
vectorized frequency grids.


RKHS and algorithm-lineage checks
---------------------------------

The Pick matrix is also documented and tested as a finite RKHS Gram matrix:

.. math::

   P_{ij}=U_i^H K_S(z_i,z_j)U_j,
   \qquad
   K_S(z,w)=\frac{I-S(z)^HS(w)}{1-\overline z w}.

This is the same positivity principle used in tangential Schur interpolation,
but the package stops at finite certificates and elementary factors.  The test
suite therefore verifies the algebraic invariants needed for a credible
baseline; it does not test a recursive chart algorithm because no such algorithm
is claimed in this release.

In particular, the Hanzon--Olivi--Peeters/Marmorat algorithms use recursive
linear-fractional Schur steps, chart data, and balanced state-space realizations
to parametrize manifolds of fixed-degree stable all-pass/lossless systems.  The
``lattice-dsp`` tangential-Schur layer is intentionally narrower: it checks Pick
positivity, constant-compatible data, diagonal/scalar reductions, and elementary
Potapov--Blaschke J-inner identities.

What is deliberately not tested as solved
-----------------------------------------

The verification suite does **not** claim a full tangential-Schur synthesis
algorithm for arbitrary feasible data.  In particular, it does not claim:

* generalized indefinite Schur interpolation;
* singular Pick-matrix parametrization;
* boundary interpolation;
* automatic construction of all rational Schur solutions;
* a complete linear-fractional parametrization of the solution set.

Those are natural future directions, but this release treats them as out of
scope.  The tested baseline is the finite definite Pick/J-inner layer that
supports the package's MIMO lattice diagnostics.

Reproducibility and tolerance policy
------------------------------------

The deep tests use deterministic random seeds and small dimensions.  Tolerances
are chosen around double-precision residuals, not benchmark speed.  Near-boundary
cases are included to expose conditioning rather than to guarantee uniform
accuracy for nearly singular interpolation problems.

Related pages
-------------

* :doc:`tangential_schur`
* :doc:`../theory/tangential_schur`
* :doc:`../examples/generated/tangential_schur_pick_jinner`
* :doc:`../examples/generated/diagonal_tangential_schur_equals_scalar`
* :doc:`../benchmarks/generated/tangential_schur_mimo`
