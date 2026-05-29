Finite-Hankel and model-reduction API
=====================================

This page collects the public model-reduction entry points.  The naming is
intentional:

* ``finite_hankel_reduce_*`` means a finite-section Hankel/Ho--Kalman-style
  reduction;
* ``finite_nehari_*`` means a finite-dimensional Nehari/AAK teaching or
  validation helper;
* ``finite_aak_siso_certificate`` means a finite-section Schmidt-pair certificate;
* exact infinite-dimensional AAK/Nehari solvers are outside the current public API scope.

SISO finite-Hankel reduction
----------------------------

.. autofunction:: lattice_dsp.iir_impulse_response
.. autofunction:: lattice_dsp.hankel_singular_values
.. autofunction:: lattice_dsp.finite_hankel_reduce_impulse
.. autofunction:: lattice_dsp.finite_hankel_reduce_iir

The compatibility names ``finite_hankel_aak_reduce_impulse`` and
``finite_hankel_aak_reduce_iir`` remain available as deprecated aliases, but new code uses the shorter ``finite_hankel_reduce_*`` names because the implementation is a finite Hankel/Ho--Kalman baseline rather than an exact AAK solver.

MIMO block-Hankel reduction
---------------------------

.. autofunction:: lattice_dsp.finite_hankel_reduce_mimo
.. autofunction:: lattice_dsp.mimo_state_space_markov_response
.. autofunction:: lattice_dsp.mimo_state_space_process_batch

The MIMO reducer accepts Markov parameters with shape ``(samples, outputs,
inputs)`` and returns state-space matrices ``A, B, C, D``.  It does not attempt
to force a scalar numerator/denominator representation onto a MIMO system.
The diagonal and coupled MIMO tutorials validate this baseline within the current MIMO scope.  ``mimo_state_space_process_batch`` is a compiled
C++/OpenMP helper for repeated simulation of the returned state-space models; it
is a runtime kernel, not a reduction algorithm.  The coupled MIMO benchmark
therefore reports processing speedup separately from one-shot and amortized
end-to-end speedup.

Finite Nehari/rational workflow
-------------------------------

.. autoclass:: lattice_dsp.FiniteNehariCandidateCriteria
   :members:

.. autofunction:: lattice_dsp.finite_nehari_approximate_tail
.. autofunction:: lattice_dsp.finite_hankel_matrix_from_tail
.. autofunction:: lattice_dsp.finite_aak_siso_certificate
.. autofunction:: lattice_dsp.finite_aak_reduce_tail
.. autofunction:: lattice_dsp.finite_aak_reduce_iir
.. autofunction:: lattice_dsp.finite_nehari_rational_candidates
.. autofunction:: lattice_dsp.select_finite_nehari_candidate
.. autofunction:: lattice_dsp.fit_rational_tail
.. autofunction:: lattice_dsp.rational_tail_response
.. autofunction:: lattice_dsp.relative_error

Current naming and stability policy
-----------------------------------

The public names are deliberately conservative.  Use ``finite_hankel_reduce_*``
for the supported finite-Hankel/Ho--Kalman baselines.  Use
``finite_aak_*`` for finite-section AAK/Nehari candidate and certificate
workflows.  These finite-section helpers are useful and tested; exact infinite-dimensional AAK/Nehari optimality is outside their scope.

The practical benchmark result is also conservative: finite-Hankel is currently
the fastest baseline in many high-order examples, while finite-section AAK/Nehari
candidates provide rank-selection and certification machinery that is valuable
for diagnostics and selected high-quality reductions.

Tutorials and validation cases
------------------------------

The recommended reading order is:

#. :doc:`../examples/generated/finite_hankel_model_reduction`
#. :doc:`../examples/generated/mimo_finite_hankel_model_reduction`
#. :doc:`../examples/generated/mimo_coupled_model_reduction`
#. :doc:`../benchmarks/generated/mimo_hankel_reduction_speedup`
#. :doc:`../examples/generated/nehari_aak_siso_toy`
#. :doc:`../benchmarks/generated/finite_nehari_rank_sweep`
#. :doc:`../examples/generated/finite_nehari_rational_bridge`
#. :doc:`../examples/generated/aak_siso_schmidt_pair_demo`
#. :doc:`../examples/generated/aak_siso_candidate_selection`
#. :doc:`../examples/generated/aak_siso_certificate_demo`
#. :doc:`../examples/generated/finite_nehari_exact_rational_tail`
#. :doc:`../examples/generated/finite_aak_noisy_tail_demo`
#. :doc:`../examples/generated/finite_aak_iir_reduction_demo`
