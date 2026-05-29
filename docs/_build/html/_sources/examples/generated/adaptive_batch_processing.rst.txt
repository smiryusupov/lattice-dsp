Batched adaptive lattice trials
===============================

.. admonition:: Tutorial goal

   Run independent adaptive system-identification trials through the batch API.

.. note::

   New to the terminology? See the :doc:`lattice DSP concept map <../../algorithms/concept_map>` and the :doc:`causality/data-use guide <../../theory/causality_and_data_use>` for how online, offline, block, and MIMO examples should be read.

Context
-------

Parameter sweeps and Monte Carlo trials are easier when many independent filters can be run
together.  This example mirrors the scalar adaptive API but uses a batch-oriented entry
point.

Key idea and equations
----------------------

Each batch member has its own input, desired signal, and final state.  The jobs are
independent, so they can be parallelized safely.

How to read the result
----------------------

Confirm that each batch member converges and that the reported batch dimensions match the requested number of trials.

Run command
-----------

.. code-block:: bash

   python examples/adaptive_batch_processing.py

Run status
----------

Return code: ``0``

Captured stdout
---------------

.. code-block:: text

   initial MSE: 0.02824192672167715
   final MSE: 0.0006448983792747414
   mean final reflection: [ 0.23467429 -0.2368577 ]
   mean final taps: [ 0.49709591 -0.20342426  0.11425937]

Source code
-----------

.. literalinclude:: ../../../examples/adaptive_batch_processing.py
   :language: python
   :linenos:
