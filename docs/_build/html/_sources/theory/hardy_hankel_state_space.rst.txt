Hardy, Hankel, reachability, and observability
==============================================

This page gives the minimum theory background behind the finite-Hankel,
Nehari/AAK, and state-space examples in ``lattice-dsp``.  It is written as an
engineering map: the definitions are included only to explain what the package
computes and what the diagnostics mean.

Causal stable systems as Hardy-space objects
--------------------------------------------

For a causal discrete-time stable SISO system, the transfer function has a power
series in nonnegative delays,

.. math::

   G(z) = g_0 + g_1 z^{-1} + g_2 z^{-2} + \cdots .

A Hardy-space viewpoint treats stable causal transfer functions as analytic
objects on the unit disk.  Informally:

* ``H^2`` tracks square-summable impulse responses and energy-style questions;
* ``H^\infty`` tracks bounded frequency responses and worst-case gain questions;
* causal projection keeps the nonnegative-delay part of a Laurent series;
* anticausal projection keeps the negative-time part that cannot be implemented
  directly by a causal stable filter.

This is the background for Nehari approximation: the distance from an anticausal
object to the space of causal stable systems is controlled by an associated
Hankel operator.  In the package, the exposed helpers are finite-section
numerical diagnostics around this picture.

Hankel operators in DSP terms
-----------------------------

Given an impulse response ``g_0, g_1, g_2, ...``, a finite Hankel matrix is

.. math::

   H_G =
   \begin{bmatrix}
   g_1 & g_2 & g_3 & \cdots \\
   g_2 & g_3 & g_4 & \cdots \\
   g_3 & g_4 & g_5 & \cdots \\
   \vdots & \vdots & \vdots & \ddots
   \end{bmatrix}.

The key interpretation is:

   A Hankel map sends past inputs to future outputs.

Large Hankel singular values correspond to input-output memory directions that
are both easy to excite and easy to observe.  Small singular values correspond to
directions that have little effect on the measured input-output behavior.  This
is why ``hankel_singular_values`` is an order-selection diagnostic.

Reachability and observability
------------------------------

A finite-dimensional state-space model is

.. math::

   x_{n+1} = A x_n + B u_n, \qquad
   y_n = C x_n + D u_n.

The reachability matrix is

.. math::

   \mathcal R = [B, AB, A^2B, \ldots, A^{n-1}B],

and the observability matrix is

.. math::

   \mathcal O =
   \begin{bmatrix}
   C \\
   CA \\
   CA^2 \\
   \vdots \\
   CA^{n-1}
   \end{bmatrix}.

A state direction is **reachable** when some input can excite it.  It is
**observable** when it can be seen in the output.  A realization is **minimal**
when every state direction is both reachable and observable.  Nonminimal states
can exist in the algebraic model while having no effect on the input-output map.

Finite Gramians provide energy-weighted versions of the same ideas:

.. math::

   W_c = \sum_{k=0}^{T-1} A^k B B^* (A^*)^k,
   \qquad
   W_o = \sum_{k=0}^{T-1} (A^*)^k C^* C A^k.

``W_c`` measures how easily the state is reached by inputs.  ``W_o`` measures
how strongly state energy appears at the output.  Hankel singular values combine
both effects.

How this connects to Ho--Kalman reduction
-----------------------------------------

The Markov parameters of a state-space model are

.. math::

   M_0 = D, \qquad M_k = C A^{k-1} B \quad (k \ge 1).

A block-Hankel matrix built from ``M_k`` factors through reachability and
observability:

.. math::

   \mathcal H =
   \begin{bmatrix}
   M_1 & M_2 & \cdots \\
   M_2 & M_3 & \cdots \\
   \vdots & \vdots & \ddots
   \end{bmatrix}
   =
   \mathcal O\,\mathcal R.

This factorization explains the finite-Hankel/Ho--Kalman workflow used by the
package:

#. compute impulse response or Markov parameters;
#. build finite Hankel matrices;
#. keep the leading singular directions;
#. construct a reduced state-space realization;
#. compare response error, stability, and runtime.

For SISO IIR reduction, the result can be converted back to scalar numerator and
denominator coefficients, and when stable, to reflection/lattice coordinates.
For MIMO reduction, the result is a reduced state-space model.

How this connects to Nehari and AAK language
--------------------------------------------

Nehari and AAK theory use Hankel operators to make statements about best stable
or low-rank rational approximation.  The package uses that theory as a guide,
but the public APIs expose finite numerical constructions:

* finite Hankel singular-value diagnostics;
* finite-Hankel/Ho--Kalman SISO and MIMO reduction;
* finite anticausal-tail approximation helpers;
* finite Schmidt-pair checks;
* rational candidate selection under explicit tolerances.

The distinction matters.  A finite matrix SVD can be an excellent diagnostic and
baseline, but it is not the same mathematical object as an exact
infinite-dimensional AAK/Nehari solver.

Practical reading checklist
---------------------------

When reading a model-reduction result, check these quantities in order:

#. **Hankel singular-value decay**: is there a clear order cutoff?
#. **Reachability/observability rank**: is the system minimal at the scale of the
   computation?
#. **Response error**: does the reduced model preserve the impulse or frequency
   response in the region that matters?
#. **Stability**: are the poles inside the unit disk, or are reflection
   coefficients available?
#. **Amortization**: does the reduced runtime pay for the one-time reduction
   cost over the expected signal length or batch count?

Related tutorials
-----------------

* :doc:`../examples/generated/reachability_observability_hankel_demo`
* :doc:`../examples/generated/finite_hankel_model_reduction`
* :doc:`../examples/generated/nehari_aak_siso_toy`
* :doc:`../examples/generated/mimo_coupled_model_reduction`
