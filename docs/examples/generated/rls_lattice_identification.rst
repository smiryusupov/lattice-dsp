RLS-style lattice-ladder identification
=======================================

.. admonition:: Tutorial goal

   Compare RLS-style adaptation with NLMS on a small stable identification problem.

.. note::

   New to the terminology? See the :doc:`lattice DSP concept map <../../algorithms/concept_map>` and the :doc:`causality/data-use guide <../../theory/causality_and_data_use>` for how online, offline, block, and MIMO examples should be read.

Context
-------

RLS updates can converge faster than NLMS when the input is correlated, at the cost of more
state and computation.  This example keeps the denominator stable and focuses on the adaptive
numerator/tap behavior.

Key idea and equations
----------------------

RLS maintains an inverse covariance estimate ``P`` and uses a gain vector that depends on the
current regressor and forgetting factor.

How to read the result
----------------------

Compare the final errors and convergence behavior for the RLS and NLMS paths.

Run command
-----------

.. code-block:: bash

   python examples/rls_lattice_identification.py

Run status
----------

Return code: ``0``

Captured stdout
---------------

.. code-block:: text

   target taps: [ 0.25 -0.15  0.65]
   NLMS taps:   [ 0.25 -0.15  0.65]
   RLS taps:    [ 0.25 -0.15  0.65]
   tail MSE NLMS: 7.570878759634834e-33
   tail MSE RLS:  1.7671936634072757e-23

Source code
-----------

.. literalinclude:: ../../../examples/rls_lattice_identification.py
   :language: python
   :linenos:
