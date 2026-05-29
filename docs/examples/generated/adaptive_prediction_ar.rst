Adaptive one-step AR prediction
===============================

.. admonition:: Tutorial goal

   Use a stable adaptive recursive model for one-step signal prediction.

.. note::

   New to the terminology? See the :doc:`lattice DSP concept map <../../algorithms/concept_map>` and the :doc:`causality/data-use guide <../../theory/causality_and_data_use>` for how online, offline, block, and MIMO examples should be read.

Context
-------

AR prediction is the scalar setting where lattice filters are historically very natural.
The model predicts the current sample from past samples while preserving a stable recursive
parameterization.

Key idea and equations
----------------------

A one-step predictor estimates

.. math::

   \hat{x}[n] = g(x[n-1], x[n-2], \ldots),\qquad e[n]=x[n]-\hat{x}[n].

How to read the result
----------------------

The final prediction error should be smaller than the initial error after the predictor adapts.

Run command
-----------

.. code-block:: bash

   python examples/adaptive_prediction_ar.py

Run status
----------

Return code: ``0``

Captured stdout
---------------

.. code-block:: text

   learned reflection: [0.211, 0.0431]
   learned numerator: [-0.5415, 0.1499, 0.1625]
   initial prediction MSE: 0.9473323949351461
   final prediction MSE: 0.10241638741994595

Source code
-----------

.. literalinclude:: ../../../examples/adaptive_prediction_ar.py
   :language: python
   :linenos:
