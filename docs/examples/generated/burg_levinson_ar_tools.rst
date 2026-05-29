Burg and Levinson-Durbin AR tools
=================================

.. admonition:: Tutorial goal

   Estimate AR coefficients using autocorrelation/Levinson and Burg-style recursions.

.. note::

   New to the terminology? See the :doc:`lattice DSP concept map <../../algorithms/concept_map>` and the :doc:`causality/data-use guide <../../theory/causality_and_data_use>` for how online, offline, block, and MIMO examples should be read.

Context
-------

This tutorial compares two common AR estimation routes.  Levinson-Durbin solves the
Yule-Walker equations from autocorrelations.  Burg estimates reflection coefficients from
forward/backward prediction errors.

Key idea and equations
----------------------

The AR model is

.. math::

   x[n]+a_1x[n-1]+\cdots+a_px[n-p]=e[n].

How to read the result
----------------------

Compare the estimated coefficients with the known synthetic model and check the final prediction error values.

Run command
-----------

.. code-block:: bash

   python examples/burg_levinson_ar_tools.py

Run status
----------

Return code: ``0``

Captured stdout
---------------

.. code-block:: text

   true reflection:      [ 0.62 -0.34  0.18 -0.08]
   Levinson reflection:  [ 0.6144 -0.3415  0.1843 -0.0766]
   Burg reflection:      [ 0.6144 -0.3416  0.1843 -0.0766]
   Burg denominator:     [ 1.      0.3274 -0.2466  0.1581 -0.0766]

Source code
-----------

.. literalinclude:: ../../../examples/burg_levinson_ar_tools.py
   :language: python
   :linenos:
