Finite Nehari / rational-candidate API
======================================

These helpers are finite-dimensional teaching and validation tools for SISO
Nehari/AAK-style workflows.  They sit above the C++ finite-Hankel primitives and
is not a full infinite-dimensional AAK or Nehari solver.

Typical workflow
----------------

.. code-block:: python

   import lattice_dsp as ld

   criteria = ld.FiniteNehariCandidateCriteria(
       max_tail_error=1e-3,
       max_rational_error=1e-2,
       max_pole_radius=0.99,
   )

   candidates = ld.finite_nehari_rational_candidates(
       tail,
       ranks=[1, 2, 3, 4],
       rows=48,
       cols=48,
       criteria=criteria,
   )
   selected = ld.select_finite_nehari_candidate(candidates)

The returned candidate rows include the finite Nehari diagnostics, fitted
rational denominator/numerator, poles, rational tail response, and an
``accepted`` flag.

High-level finite-section reducer
---------------------------------

``finite_aak_reduce_tail`` wraps the candidate workflow and optionally attaches
a finite Schmidt-pair certificate for the selected rank.  This is the current
high-level finite-section SISO AAK/Nehari workflow.  It is useful for tutorials
and controlled experiments, but it is still not a full infinite-dimensional
Hardy-space solver.

.. code-block:: python

   result = ld.finite_aak_reduce_tail(
       tail,
       ranks=[1, 2, 3, 4, 5, 6],
       rows=48,
       cols=48,
       criteria=criteria,
   )

   print(result["selected_rank"])
   print(result["selected"]["poles"])
   print(result["certificate"]["schmidt_left_residual"])


For an actual stable lattice/IIR model, use ``finite_aak_reduce_iir``.  It computes
an impulse response internally, runs the same finite-section candidate workflow,
and returns the reduced numerator, denominator, and reflection coefficients when
stable.

.. code-block:: python

   iir_result = ld.finite_aak_reduce_iir(
       reflection,
       numerator,
       ranks=[2, 3, 4, 5, 6, 8],
       n_impulse=192,
       rows=96,
       cols=96,
       criteria=criteria,
   )

   print(iir_result["selected_rank"])
   print(iir_result["reduced_reflection"])
   print(iir_result["relative_impulse_error"])


Finite AAK/Nehari certificate
-----------------------------

For a more AAK-shaped diagnostic, use
``finite_aak_siso_certificate``.  It computes the finite Hankel SVD for a chosen
target rank, reports the first neglected singular value, verifies the finite
Schmidt-pair equations, and attaches the rational candidate for the same rank.

.. code-block:: python

   certificate = ld.finite_aak_siso_certificate(
       tail,
       rank=3,
       rows=48,
       cols=48,
       criteria=criteria,
   )

   print(certificate["sigma_next"])
   print(certificate["schmidt_left_residual"])
   print(certificate["schmidt_right_residual"])
   print(certificate["candidate"]["poles"])

This is still a finite-section certificate rather than a full infinite-dimensional
Hardy-space AAK/Nehari solver, but it makes the singular-vector target and the
rational candidate explicit.

Exact validation case
---------------------

A tail of the form

.. math::

   \gamma_n=\sum_{i=1}^r c_i p_i^n,\qquad |p_i|<1,

has finite Hankel rank at most ``r``.  The
``finite_nehari_exact_rational_tail.py`` tutorial uses this fact as a controlled
validation: ranks below the known order are rejected, while the correct rank
recovers the poles and tail to near numerical precision.

Reference
---------

.. automodule:: lattice_dsp.nehari
   :members:
   :undoc-members:
   :show-inheritance:
