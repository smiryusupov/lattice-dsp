Large echo-scale recursive model stress
=======================================

.. admonition:: Tutorial goal

   Stress a million-sample signal with a high-order stable lattice-ladder model and compare the scale with a long FIR echo tap vector.

.. note::

   New to the terminology? See the :doc:`lattice DSP concept map <../../algorithms/concept_map>` and the :doc:`causality/data-use guide <../../theory/causality_and_data_use>` for how online, offline, block, and MIMO examples should be read.

Context
-------

Echo-style paths often have long memory: delay, early reflections, and a slowly decaying
room tail.  FIR adaptive echo cancellers model that memory by adding taps, so a long path
becomes a large parameter vector that is filtered and updated at every sample.  This example
keeps adaptation out of scope and instead stresses the fixed-filter processing axis: a
million-sample input and a high-order stable recursive lattice-ladder model.

Key idea and equations
----------------------

A direct FIR echo model with ``L`` taps has the filtering relation

.. math::

   y[n] = \sum_{m=0}^{L-1} h[m] x[n-m],

and an LMS-style update touches the same large tap vector again,

.. math::

   h_{n+1} = h_n + \mu e[n] x_n.

A lattice-ladder IIR model stores recursive state and stage parameters.  Its scalar
all-pole stability guard is still expressed through bounded reflection coefficients,

.. math::

   |k_i| < 1.

The comparison is a scale diagnostic: ``N L`` direct FIR tap visits versus ``N p``
lattice-stage visits for recursive order ``p``.  It is not an accuracy-equivalence claim.

How to read the result
----------------------

Compare the local lattice-ladder timing with the printed FIR echo-scale tap-visit estimates, especially the FIR taps / lattice order ratio.

Run command
-----------

.. code-block:: bash

   python examples/large_order_echo_stress.py

Run status
----------

Return code: ``0``

Captured stdout
---------------

.. code-block:: text

   large echo-scale stable recursive model stress
   ========================================================
   samples: 1,000,000
   lattice-ladder recursive order: 512
   max |reflection|: 0.372387
   recursive state count: 512
   ladder parameter count: 513
   median IIR/lattice-ladder time: 0.775717 s
   throughput: 1.29 million samples/s
   stage update rate: 0.66 billion stage-visits/s
   output RMS: 3.107450
   
   echo-scale comparison numbers
   --------------------------------------------------------
   reference FIR echo taps: 131,072
   FIR taps / lattice order: 256.0x
   lattice stage visits: 512,000,000
   FIR filter tap visits, direct form: 131,072,000,000
   FIR LMS filter+update tap visits, rough scale: 262,144,000,000
   note: the tap-visit numbers are scale diagnostics, not an accuracy equivalence claim

Generated data files
--------------------

* :download:`large_order_echo_stress.csv <_artifacts/large_order_echo_stress/large_order_echo_stress.csv>`

Source code
-----------

.. literalinclude:: ../../../examples/large_order_echo_stress.py
   :language: python
   :linenos:
