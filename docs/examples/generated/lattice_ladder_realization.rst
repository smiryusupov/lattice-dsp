Lattice-ladder realization versus direct numerator coefficients
===============================================================

.. admonition:: Tutorial goal

   Show that lattice-ladder taps can realize the same transfer function as direct numerator coefficients.

.. note::

   New to the terminology? See the :doc:`lattice DSP concept map <../../algorithms/concept_map>` and the :doc:`causality/data-use guide <../../theory/causality_and_data_use>` for how online, offline, block, and MIMO examples should be read.

Context
-------

A lattice filter gives a stable recursive denominator.  Ladder taps then combine the
internal lattice states to realize a numerator.  This example is a sanity check that the
lattice-ladder representation and a direct numerator representation agree sample-by-sample.

Key idea and equations
----------------------

The transfer function is still

.. math::

   H(z)=\frac{B(z)}{A(z)},

but ``A`` is controlled by reflection coefficients while ``B`` is represented by ladder
taps.

How to read the result
----------------------

The important number is the maximum absolute output difference; it should be close to floating-point roundoff.

Run command
-----------

.. code-block:: bash

   python examples/lattice_ladder_realization.py

Run status
----------

Return code: ``0``

Captured stdout
---------------

.. code-block:: text

   reflection: [0.35, -0.2, 0.1]
   direct numerator: [0.2, -0.1, 0.05, 0.75]
   ladder taps: [0.07164000000000001, 0.06959999999999998, -0.14499999999999996, 0.75]
   lattice numerator reconstruction: [0.2, -0.1, 0.04999999999999999, 0.75]
   max |direct - lattice|: 8.882e-16

Source code
-----------

.. literalinclude:: ../../../examples/lattice_ladder_realization.py
   :language: python
   :linenos:
