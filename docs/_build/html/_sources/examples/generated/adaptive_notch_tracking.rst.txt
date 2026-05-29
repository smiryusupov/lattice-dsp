Adaptive notch tracking
=======================

.. admonition:: Tutorial goal

   Track and suppress a sinusoidal interferer with a stable second-order notch model.

.. note::

   New to the terminology? See the :doc:`lattice DSP concept map <../../algorithms/concept_map>` and the :doc:`causality/data-use guide <../../theory/causality_and_data_use>` for how online, offline, block, and MIMO examples should be read.

Context
-------

Notch filters are a compact way to remove narrowband interference.  This tutorial uses a
small adaptive example to show how a stable recursive structure can follow an interfering
tone.

Key idea and equations
----------------------

A second-order notch has a pair of zeros near the interference frequency and stable poles
whose radius controls bandwidth.

How to read the result
----------------------

Inspect the estimated notch frequency and the before/after error or suppression metric.

Run command
-----------

.. code-block:: bash

   python examples/adaptive_notch_tracking.py

Run status
----------

Return code: ``0``

Captured stdout
---------------

.. code-block:: text

   true frequency [Hz]:      1240.0
   estimated frequency [Hz]: 1238.68
   input RMS:                0.7088300560372145
   output RMS:               0.12923913178119548

Source code
-----------

.. literalinclude:: ../../../examples/adaptive_notch_tracking.py
   :language: python
   :linenos:
