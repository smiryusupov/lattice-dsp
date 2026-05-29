Streaming block processing equivalence
======================================

.. admonition:: Tutorial goal

   Show that stateful block processing matches one-shot processing.

.. note::

   New to the terminology? See the :doc:`lattice DSP concept map <../../algorithms/concept_map>` and the :doc:`causality/data-use guide <../../theory/causality_and_data_use>` for how online, offline, block, and MIMO examples should be read.

Context
-------

Real-time DSP usually processes blocks rather than entire arrays.  This tutorial checks that
block boundaries do not change the result when filter state is carried correctly.

Key idea and equations
----------------------

For a stateful recursion, the output over concatenated blocks should match the output over
the full signal when final state from one block is used as initial state for the next.

How to read the result
----------------------

The reported maximum difference should be close to roundoff.

Run command
-----------

.. code-block:: bash

   python examples/streaming_block_processing.py

Run status
----------

Return code: ``0``

Captured stdout
---------------

.. code-block:: text

   streaming matches one-shot: True
   RLS streaming tail MSE: 2.269267880262604e-21
   RLS learned taps: [0.3  0.   0.55]

Source code
-----------

.. literalinclude:: ../../../examples/streaming_block_processing.py
   :language: python
   :linenos:
