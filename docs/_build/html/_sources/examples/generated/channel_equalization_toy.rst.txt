Toy channel equalization
========================

.. admonition:: Tutorial goal

   Small fixed-denominator equalization/system-identification prototype.

.. note::

   New to the terminology? See the :doc:`lattice DSP concept map <../../algorithms/concept_map>` and the :doc:`causality/data-use guide <../../theory/causality_and_data_use>` for how online, offline, block, and MIMO examples should be read.

Context
-------

This example is a compact equalization-style demo.  It keeps the setting small so readers can
inspect the signal, filter, and error terms without needing a communication-system framework.

Key idea and equations
----------------------

Equalization tries to learn an inverse or compensating filter so the output approaches a
desired reference signal.

How to read the result
----------------------

Inspect the printed MSE or improvement metric and remember this is a toy diagnostic, not a modem benchmark.

Run command
-----------

.. code-block:: bash

   python examples/channel_equalization_toy.py

Run status
----------

Return code: ``0``

Captured stdout
---------------

.. code-block:: text

   estimated equalizer numerator: [1.25116, 0.29751, -0.46007]
   initial MSE: 0.04211684200241321
   final MSE: 0.0010456849376973291

Source code
-----------

.. literalinclude:: ../../../examples/channel_equalization_toy.py
   :language: python
   :linenos:
