LMS as a minimax robust filter
================================

This page gives the conceptual background for the tutorial
``examples/hinf_lms_reproduction.py``.  The purpose is not to replace the
original paper, but to make one of its most useful messages visible in a small
reproducible experiment.

The historical surprise
-----------------------

LMS is often introduced as a cheap stochastic-gradient approximation to a
least-squares problem.  RLS, by contrast, is presented as the recursion that
solves a recursive least-squares objective exactly.  That story is correct, but
it is incomplete.

The historical timing makes the result memorable.  The LMS/delta-rule idea goes
back to Widrow and Hoff's 1960 adaptive switching work.  For more than three
decades, LMS was mainly taught as the simple, inexpensive member of the
least-squares family: useful, robust in practice, but approximate.  Hassibi,
Sayed, and Kailath then showed in their 1996 paper *H∞ Optimality of the LMS
Algorithm* that LMS also has a deterministic robust-filtering interpretation.

From this viewpoint, LMS is tied to an H∞ minimax problem: it controls a
worst-case energy gain from disturbances to estimation errors.  The same
recursion can therefore look like a crude least-squares approximation from one
angle, and like a central robust filter from another.  That is the flagship
lesson of the tutorial: the algorithm did not change, but the performance lens
did.

This also connects to a practical point emphasized in classical adaptive-filter
texts, including Odile Macchi's LMS-focused treatment: even FIR LMS is not
"automatic."  The learning rate controls convergence speed, misadjustment, and
divergence.  Recursive/IIR adaptation keeps that step-size problem and adds a
structural one: denominator updates move the poles of the filter.

Adaptive-filter model
---------------------

The tutorial uses the standard linear adaptive-filtering model

.. math::

   d_i = u_i^T w_\star + v_i,

where ``u_i`` is the input/regressor, ``w_*`` is the unknown parameter vector,
and ``v_i`` is a disturbance.  The estimator produces ``w_i`` and the prediction
error is

.. math::

   e_i = d_i - u_i^T w_i.

The least-squares viewpoint asks for small accumulated squared error, often
under statistical assumptions on the disturbance.  The robust viewpoint asks a
different question: how much can an arbitrary disturbance sequence amplify the
error sequence?

Finite-horizon gain diagnostic
------------------------------

For fixed regressors and a fixed adaptive algorithm, the map from disturbance
samples to noise-induced prediction errors is linear after subtracting the
zero-disturbance trajectory.  The tutorial forms this finite-horizon sensitivity
matrix ``M`` numerically and estimates

.. math::

   \gamma^2 = \sup_{v\ne0}\frac{\|M v\|_2^2}{\|v\|_2^2}
            = \sigma_{\max}(M)^2.

The right singular vector associated with ``σ_max`` is the finite-horizon
worst-case disturbance direction for that algorithm and regressor sequence.
This is not a full proof of H∞ optimality.  It is a visual diagnostic that makes
the H∞ question concrete.

How to read the tutorial
------------------------

The generated page compares LMS, NLMS, and RLS in two ways.

* Under benign random disturbance, RLS usually converges fastest.  This is the
  classical least-squares story.
* Under a worst-case disturbance direction, an estimator that looks excellent
  on random noise can show a larger disturbance-to-error gain.  This is the
  minimax robustness story.

The point is not that LMS is always better than RLS.  The point is that the
algorithmic objective being emphasized has changed.  Average-case and worst-case
views can rank algorithms differently.

Connection to lattice filters
-----------------------------

This package focuses on stable recursive DSP.  The H∞ tutorial belongs here
because it explains another form of robustness.  Lattice parameterizations
control stability by working in reflection-coefficient coordinates; the
H∞ viewpoint controls worst-case disturbance amplification.  Both are part of
the same practical theme: recursive adaptive filters should not be judged only
by their behavior under friendly stochastic assumptions.

Run the tutorial
----------------

Build the generated Sphinx tutorial pages:

.. code-block:: bash

   ./scripts/build_docs_with_results.sh

Then open the examples gallery and select **LMS through the H-infinity lens**:

.. code-block:: bash

   xdg-open docs/_build/html/examples/index.html

References
----------

* B. Widrow and M. E. Hoff, Jr., "Adaptive Switching Circuits," 1960 IRE
  WESCON Convention Record, 1960.
* Odile Macchi, *Adaptive Processing: The Least Mean Squares Approach with
  Applications in Transmission*, Wiley, 1995.
* B. Hassibi, A. H. Sayed, and T. Kailath, "H∞ Optimality of the LMS
  Algorithm," *IEEE Transactions on Signal Processing*, 44(2), 1996.
