Reflection coefficients and denominator coefficients
====================================================

.. admonition:: Tutorial goal

   Convert between reflection/PARCOR coefficients and a conventional IIR denominator.

.. note::

   New to the terminology? See the :doc:`lattice DSP concept map <../../algorithms/concept_map>` and the :doc:`causality/data-use guide <../../theory/causality_and_data_use>` for how online, offline, block, and MIMO examples should be read.

Context
-------

This is the shortest path into the package.  A stable all-pole IIR filter can be
represented either by denominator coefficients or by reflection coefficients.  The
reflection form is more convenient for adaptive work because stability is controlled by
simple per-stage bounds.

Key idea and equations
----------------------

For an all-pole denominator

.. math::

   A(z)=1+a_1z^{-1}+\cdots+a_pz^{-p},

scalar lattice stability is guaranteed when every reflection coefficient satisfies

.. math::

   |k_i| < 1.

How to read the result
----------------------

Check that converting reflection coefficients to a denominator and back returns the original values up to numerical precision.

Run command
-----------

.. code-block:: bash

   python examples/reflection_conversion.py

Run status
----------

Return code: ``0``

Captured stdout
---------------

.. code-block:: text

   reflection: [0.7, -0.4, 0.25]
   denominator: [1.0, 0.32, -0.295, 0.25]
   restored: [0.7, -0.4, 0.25]
   max pole radius: 0.9280944783322326

Source code
-----------

.. literalinclude:: ../../../examples/reflection_conversion.py
   :language: python
   :linenos:
