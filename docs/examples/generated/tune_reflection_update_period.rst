Tuning the reflection update period
===================================

.. admonition:: Tutorial goal

   Explore the speed/quality tradeoff from updating reflection coefficients less frequently.

.. note::

   New to the terminology? See the :doc:`lattice DSP concept map <../../algorithms/concept_map>` and the :doc:`causality/data-use guide <../../theory/causality_and_data_use>` for how online, offline, block, and MIMO examples should be read.

Context
-------

Updating denominator parameters can be more expensive than updating numerator taps.  This
tutorial sweeps the reflection update period and reports which periods preserve quality while
reducing work.

Key idea and equations
----------------------

A period ``P`` updates reflection coefficients only when

.. math::

   n \equiv 0 \pmod P.

How to read the result
----------------------

Look for the largest period that keeps the tail MSE close to the period-1 baseline.

Run command
-----------

.. code-block:: bash

   python examples/tune_reflection_update_period.py

Run status
----------

Return code: ``0``

Captured stdout
---------------

.. code-block:: text

   recommended period: 32
   scope: single_signal
   warnings: ['Only one/few validation trial(s) were provided, so the selected reflection_update_period is signal-specific rather than a robust default. Pass a 2-D trial-by-sample array to validate worst-case behavior across independent signals.']
   recommended row: {'reflection_update_period': 32, 'n_trials': 1, 'min_s': 0.0017357261385768652, 'median_s': 0.0017357261385768652, 'max_s': 0.0017357261385768652, 'speedup_vs_period1_median': 11.66822032312495, 'mse_tail_median': 0.0011291399662327094, 'mse_tail_worst': 0.0011291399662327094, 'mse_total_median': 0.031157863250376363, 'tail_mse_ratio_median': 1.3431185662494163, 'tail_mse_ratio_p90': 1.3431185662494163, 'tail_mse_ratio_worst': 1.3431185662494163, 'stability_margin_median': 0.6885322506418122, 'stability_margin_min': 0.6885322506418122, 'max_abs_reflection_median': 0.3114677493581878, 'max_abs_reflection_max': 0.3114677493581878}
   robust recommended period: 32
   robust scope: robust

Source code
-----------

.. literalinclude:: ../../../examples/tune_reflection_update_period.py
   :language: python
   :linenos:
