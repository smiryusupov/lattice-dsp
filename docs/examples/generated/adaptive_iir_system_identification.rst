Stable adaptive IIR system identification
=========================================

.. admonition:: Tutorial goal

   Identify a synthetic recursive system while keeping the learned denominator stable.

.. note::

   New to the terminology? See the :doc:`lattice DSP concept map <../../algorithms/concept_map>` and the :doc:`causality/data-use guide <../../theory/causality_and_data_use>` for how online, offline, block, and MIMO examples should be read.

Context
-------

This is a two-signal system-identification example, not one-step
self-prediction.  A known stable target system receives the reference
input ``x[n]`` and generates the desired signal ``d[n]``.  The adaptive
lattice-ladder model receives the same reference input and learns a
stable recursive approximation to the target input-output map.

Key idea and equations
----------------------

The data relationship is

.. math::

   d[n] = H_{\mathrm{target}}(q^{-1})x[n],
   \qquad
   \widehat d[n] = H_{\theta_n}(q^{-1})x[n].

The instantaneous error is

.. math::

   e[n] = d[n] - \widehat d[n].

The adaptive model updates numerator/ladder parameters and reflection
coefficients after forming the current output and error.  The reflection
update is bounded so the learned denominator remains stable during
training.  This is causal adaptive filtering because ``\widehat d[n]``
uses current/past reference samples and previous filter state, not future
desired samples.

Causality and data use
----------------------

This is inductive/streaming system identification on a synthetic sequence: the target generates ``d[n]`` from the same reference ``x[n]`` seen by the adaptive filter.  It is not a single-signal predictor, and it is not a production echo canceller.

How to read the result
----------------------

Compare the initial and final MSE.  Also inspect the learned reflection coefficients and pole radius if printed.

Run command
-----------

.. code-block:: bash

   python examples/adaptive_iir_system_identification.py

Run status
----------

Return code: ``0``

Captured stdout
---------------

.. code-block:: text

   target reflection: [0.45, -0.25]
   learned reflection: [0.4324, -0.2655]
   target numerator: [0.4, -0.1, 0.7]
   learned numerator: [0.4014, -0.1094, 0.7015]
   initial MSE: 0.1314096982708521
   final MSE: 0.00027527991031382895

Source code
-----------

.. literalinclude:: ../../../examples/adaptive_iir_system_identification.py
   :language: python
   :linenos:
