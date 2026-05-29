Adaptive lattice filters: NLMS and RLS
======================================

Overview
--------

``lattice-dsp`` contains adaptive examples and APIs for stable recursive system
identification.  The key idea is to update filter parameters while keeping the
recursive denominator in a stability-preserving representation.

NLMS-style updates
------------------

Normalized LMS adapts a parameter vector using the instantaneous error

.. math::

   e[n] = d[n] - y[n],

and a normalized step size

.. math::

   \theta[n+1] = \theta[n] + \mu\,\frac{e[n] x[n]}{\epsilon + \|x[n]\|^2}.

The package uses this principle for ladder/tap adaptation and for the adaptive
lattice-ladder demos.  When reflection updates are used, raw unconstrained
parameters are mapped into bounded reflection coefficients so the denominator
remains stable.

Update period scaling
---------------------

Recursive-filter adaptation can be expensive.  ``reflection_update_period`` lets
you update the reflection coefficients every ``P`` samples instead of every
sample.  ``scale_reflection_mu_by_period=True`` scales the step size by the
period so slower update schedules maintain comparable adaptation strength.  The
tuning helpers and reports demonstrate the speed/quality tradeoff.

H∞ viewpoint
-------------

The usual LMS story is that it is a stochastic-gradient approximation to a
least-squares objective.  The tutorial :doc:`robust_lms_hinf` explains the
complementary Hassibi--Sayed--Kailath viewpoint: LMS can also be read as a
minimax robust filter that controls worst-case disturbance-to-error energy
gain.  This does not replace the least-squares interpretation; it shows that
the same recursion can be understood through a different objective.

RLS-style updates
-----------------

Recursive Least Squares uses an inverse correlation estimate ``P`` and a
forgetting factor ``lambda``.  In a standard FIR form:

.. math::

   g[n] = \frac{P[n-1] x[n]}{\lambda + x[n]^T P[n-1] x[n]},

.. math::

   w[n] = w[n-1] + g[n] e[n],

.. math::

   P[n] = \lambda^{-1}\left(P[n-1] - g[n] x[n]^T P[n-1]\right).

The package exposes ``LatticeLadderRLS`` and ``rls_process_batch`` as a compact
RLS adaptive path.  It is intentionally presented as an adaptive-filtering tool,
not as a full acoustic echo canceller.

OpenMP batch path
-----------------

``adaptive_process_batch`` and ``rls_process_batch`` process independent jobs in
parallel when OpenMP is available.  This is useful for parameter sweeps, channel
batches, and repeated benchmark trials.

Relevant APIs
-------------

* ``LatticeLadderNLMS``
* ``AdaptiveLatticeLadderNLMS``
* ``LatticeLadderRLS``
* ``adaptive_process_batch``
* ``rls_process_batch``
* ``tune_reflection_update_period``

Examples
--------

* ``examples/hinf_lms_reproduction.py``
* ``examples/adaptive_iir_system_identification.py``
* ``examples/rls_lattice_identification.py``
* ``examples/tracking_drifting_iir_system.py``
* ``examples/tune_reflection_update_period.py``
