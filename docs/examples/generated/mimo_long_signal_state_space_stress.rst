MIMO long-signal state-space stress
===================================

.. admonition:: Tutorial goal

   Reduce a coupled MIMO state-space model with the finite block-Hankel workflow, then process long batched multichannel signals through the compiled C++ runtime.

.. note::

   New to the terminology? See the :doc:`lattice DSP concept map <../../algorithms/concept_map>` and the :doc:`causality/data-use guide <../../theory/causality_and_data_use>` for how online, offline, block, and MIMO examples should be read.

Context
-------

This is the multichannel counterpart to the scalar long-signal stress examples.  A
coupled MIMO system is converted to Markov parameters, reduced with
``finite_hankel_reduce_mimo``, and then reused on a long batched input through
``mimo_state_space_process_batch``.  The printed comparison numbers are scale
diagnostics for MIMO echo-style paths, not claims of accuracy equivalence to every
long FIR model and not a matrix-valued AAK/Nehari solver claim.

Key idea and equations
----------------------

A MIMO state-space model uses

.. math::

   x_s[n+1] = A x_s[n] + B u[n], \qquad
   y[n] = C x_s[n] + D u[n].

Its Markov matrices ``M_k`` map input channels to output channels at lag ``k``.
The finite MIMO block-Hankel reducer builds a matrix whose blocks are these
Markov matrices and returns a lower-order state-space realization.  For comparison,
a direct MIMO FIR echo model with ``L`` taps per input-output path has scale

.. math::

   N \, L \, m \, p,

for ``N`` samples, ``m`` inputs, and ``p`` outputs.  The state-space runtime has a
dense recursive scale tied to the chosen state order instead of the FIR tap count.

How to read the result
----------------------

Inspect the finite block-Hankel reduction time, retained energy, reduced runtime, output-channel throughput, and the printed direct MIMO FIR tap-visit scale.

Run command
-----------

.. code-block:: bash

   python examples/mimo_long_signal_state_space_stress.py

Representative local output
---------------------------

The following output is from one local run of the default 8-by-8 stress command.
Exact timings are machine-dependent, but the scale relationship is the point:
the finite MIMO reduction produced a stable order-16 model that processed one
million multichannel samples while the equivalent direct MIMO FIR tap-visit
count was orders of magnitude larger.

.. code-block:: text

   MIMO long-signal finite-Hankel/state-space stress
   ================================================================
   batch streams: 1
   samples per stream: 1,000,000
   inputs x outputs: 8 x 8
   full MIMO state order: 64
   dominant full-model pole radius target: 0.985000
   full-model spectral radius: 0.985000
   Markov samples for reduction: 320
   block-Hankel matrix: 192 x 192
   reduced order: 16
   Markov generation time: 0.004333 s
   finite MIMO block-Hankel reduction time: 0.332755 s
   retained Hankel energy: 0.999787
   relative Markov error: 1.963e-03
   reduced model stable: True
   reduced spectral radius: 0.983021

   compiled reduced MIMO runtime
   ----------------------------------------------------------------
   median reduced state-space time: 0.151889 s
   throughput: 6.58 million multichannel samples/s
   output-channel throughput: 52.67 million output samples/s
   dense reduced state-space visits: 576,000,000
   dense visit rate: 3.79 billion visits/s
   output RMS: 1.914513

   MIMO echo-scale comparison numbers
   ----------------------------------------------------------------
   reference MIMO FIR taps per input-output path: 131,072
   FIR taps / reduced state order: 8192.0x
   full dense state-space visits at same signal size: 5,184,000,000
   reduced dense state-space visits: 576,000,000
   direct MIMO FIR filter visits: 8,388,608,000,000
   direct MIMO FIR LMS filter+update visits, rough scale: 16,777,216,000,000
   note: these are scale diagnostics, not an accuracy equivalence claim

Run status
----------

Return code: ``0``

Captured stdout
---------------

.. code-block:: text

   MIMO long-signal finite-Hankel/state-space stress
   ================================================================
   batch streams: 1
   samples per stream: 1,000,000
   inputs x outputs: 8 x 8
   full MIMO state order: 64
   dominant full-model pole radius target: 0.985000
   full-model spectral radius: 0.985000
   Markov samples for reduction: 320
   block-Hankel matrix: 192 x 192
   reduced order: 16
   Markov generation time: 0.003492 s
   finite MIMO block-Hankel reduction time: 0.303075 s
   retained Hankel energy: 0.999787
   relative Markov error: 1.963e-03
   reduced model stable: True
   reduced spectral radius: 0.983021
   
   compiled reduced MIMO runtime
   ----------------------------------------------------------------
   median reduced state-space time: 0.149827 s
   throughput: 6.67 million multichannel samples/s
   output-channel throughput: 53.39 million output samples/s
   dense reduced state-space visits: 576,000,000
   dense visit rate: 3.84 billion visits/s
   output RMS: 1.914513
   
   MIMO echo-scale comparison numbers
   ----------------------------------------------------------------
   reference MIMO FIR taps per input-output path: 131,072
   FIR taps / reduced state order: 8192.0x
   full dense state-space visits at same signal size: 5,184,000,000
   reduced dense state-space visits: 576,000,000
   direct MIMO FIR filter visits: 8,388,608,000,000
   direct MIMO FIR LMS filter+update visits, rough scale: 16,777,216,000,000
   note: these are scale diagnostics, not an accuracy equivalence claim

Generated data files
--------------------

* :download:`mimo_long_signal_state_space_stress.csv <_artifacts/mimo_long_signal_state_space_stress/mimo_long_signal_state_space_stress.csv>`

Source code
-----------

.. literalinclude:: ../../../examples/mimo_long_signal_state_space_stress.py
   :language: python
   :linenos:
