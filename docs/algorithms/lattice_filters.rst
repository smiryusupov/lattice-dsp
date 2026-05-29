Scalar lattice and lattice-ladder IIR filters
=============================================

Motivation
----------

A direct-form IIR denominator

.. math::

   A(z) = 1 + a_1 z^{-1} + \cdots + a_p z^{-p}

is stable only if all roots of ``A(z)`` lie inside the unit circle.  Adaptive
updates to ``a_i`` can easily move poles outside the unit circle.  Lattice
filters use reflection coefficients, also called PARCOR coefficients, instead.
For the scalar all-pole case, the sufficient stability condition is simple:

.. math::

   |k_i| < 1 \quad \text{for every stage } i.

This is the central reason for using lattice parameterizations in this package.
The package also exposes ``bounded_reflection_from_raw`` so unconstrained raw
optimization variables can be mapped through ``tanh`` into stable reflection
coefficients.

Reflection-to-denominator recursion
-----------------------------------

The conversion implemented by ``reflection_to_denominator`` is the standard
step-up recursion.  Let ``a^(m)`` be the order-``m`` denominator.  Given
reflection coefficient ``k_m``:

.. math::

   a_i^{(m)} = a_i^{(m-1)} + k_m a_{m-i}^{(m-1)},\quad i=1,\ldots,m-1,

and

.. math::

   a_m^{(m)} = k_m.

The inverse conversion, implemented by ``denominator_to_reflection``, is the
step-down recursion.  It peels off the last reflection coefficient and reduces
the denominator order.  If any step would violate stability, the corresponding
reflection coefficient exposes that violation.

Causality and state
-------------------

The scalar runtime filters in this package are causal, stateful IIR filters.
``LatticeIIR.process_sample(x)`` and ``LatticeLadderIIR.process_sample(x)``
consume one current input sample, update an internal state, and return the
current output.  The vector ``process`` helpers simply call the same
sample-by-sample recurrence from left to right.  They do not look ahead in the
input sequence.

This is different from AR coefficient estimators such as Burg or
Levinson-Durbin.  Those routines estimate a denominator from a finite data
record or autocorrelation sequence.  Once the coefficients have been estimated,
the resulting IIR or prediction-error filter can be run causally.

Lattice-ladder realization
--------------------------

The lattice part represents the stable recursive denominator.  Ladder taps then
combine intermediate forward/backward prediction errors to realize a numerator.
The package provides conversions between conventional numerator coefficients and
ladder taps:

* ``numerator_to_ladder``
* ``ladder_to_numerator``

This makes it possible to compare the lattice-ladder output with conventional
``scipy.signal.lfilter`` while preserving the stable recursive parameterization.

Analytic Jacobians
------------------

For optimization experiments, ``lattice-dsp`` includes analytic Jacobians from
reflection/raw parameters to denominator coefficients:

* ``denominator_reflection_jacobian``
* ``denominator_raw_jacobian``
* ``denominator_raw_jacobian_finite_difference``

The finite-difference version is included as a debugging oracle for the analytic
form.

Relevant APIs
-------------

* ``LatticeIIR``
* ``LatticeLadderIIR``
* ``reflection_to_denominator``
* ``denominator_to_reflection``
* ``bounded_reflection_from_raw``
* ``numerator_to_ladder``
* ``ladder_to_numerator``

Examples
--------

* ``examples/reflection_conversion.py``
* ``examples/lattice_ladder_realization.py``
* ``examples/stability_vs_direct_iir.py``

References
----------

Classic background for lattice filters, PARCOR coefficients, and stable
all-pole modeling is covered in linear-prediction and adaptive-filtering texts;
see :doc:`../references`.
