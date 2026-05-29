OpenMP batch processing for independent streams
===============================================

.. admonition:: Tutorial goal

   Run many independent signals through the C++ backend and compare batch behavior.

.. note::

   New to the terminology? See the :doc:`lattice DSP concept map <../../algorithms/concept_map>` and the :doc:`causality/data-use guide <../../theory/causality_and_data_use>` for how online, offline, block, and MIMO examples should be read.

Context
-------

The C++ extension is most useful when the same lattice operation is applied to many
independent channels, trials, or parameter settings.  This example demonstrates the batch
interface used by the benchmarks.

Key idea and equations
----------------------

If ``C`` independent channels each contain ``N`` samples, the batch path parallelizes over
independent jobs while preserving each stream's state recursion.

How to read the result
----------------------

Check whether OpenMP is reported as available and compare the batch output against the scalar reference path.

Run command
-----------

.. code-block:: bash

   python examples/openmp_batch_processing.py

Run status
----------

Return code: ``0``

Captured stdout
---------------

.. code-block:: text

   OpenMP enabled: True
   input shape:    (64, 50000)
   output shape:   (64, 50000)
   output RMS:     0.9071251047245191

Source code
-----------

.. literalinclude:: ../../../examples/openmp_batch_processing.py
   :language: python
   :linenos:
