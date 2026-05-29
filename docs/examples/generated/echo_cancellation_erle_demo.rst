Synthetic ERLE metric demo
==========================

.. admonition:: Tutorial goal

   Explain ERLE and MSE on a controlled synthetic echo-path identification problem.

.. note::

   New to the terminology? See the :doc:`lattice DSP concept map <../../algorithms/concept_map>` and the :doc:`causality/data-use guide <../../theory/causality_and_data_use>` for how online, offline, block, and MIMO examples should be read.

Context
-------

This tutorial is intentionally limited: it is a metric demonstration for
a synthetic echo path, not a production acoustic echo canceller.  It is
useful for understanding the two-signal roles: a far-end/reference signal
drives the unknown echo path, while a microphone/desired signal is used
to measure the residual after subtracting the estimated echo.

Key idea and equations
----------------------

With reference signal :math:`x[n]` and microphone signal :math:`m[n]`, an
adaptive echo-path model produces

.. math::

   \widehat e_{echo}[n] = h_{\theta_n}(x)[n],
   \qquad
   r[n] = m[n] - \widehat e_{echo}[n].

Echo return loss enhancement is commonly reported as

.. math::

   \operatorname{ERLE}_{dB}
     =10\log_{10}\frac{\mathbb{E}[m^2]}{\mathbb{E}[r^2]}.

A causal echo-path filter may use current and past reference samples and
previous adaptive state.  It must not use future microphone samples to
form the current residual.

Causality and data use
----------------------

This is a controlled two-signal synthetic metric demo.  It uses reference/far-end and microphone/desired roles, unlike one-signal AR prediction.  Real AEC systems require additional delay control, double-talk handling, nonlinear echo handling, and deployment-specific engineering.

How to read the result
----------------------

Higher ERLE and lower MSE indicate better cancellation in this controlled synthetic setup only; they are not product-quality acoustic echo-cancellation claims.

Run command
-----------

.. code-block:: bash

   python examples/echo_cancellation_erle_demo.py

Run status
----------

Return code: ``0``

Captured stdout
---------------

.. code-block:: text

   Synthetic echo metric demo
   --------------------------
   No cancellation ERLE:     0.00 dB
   FIR/NLMS ERLE:           24.98 dB
   Lattice/IIR ERLE:        16.18 dB
   
   ERLE = 10 log10(input echo/error power / output residual error power).
   Positive ERLE means residual error power was reduced. It does not by
   itself prove speech quality, especially during double-talk.
   
   Final lattice reflection coefficients:
   [ 0.2837 -0.1909  0.1629 -0.1216]

Source code
-----------

.. literalinclude:: ../../../examples/echo_cancellation_erle_demo.py
   :language: python
   :linenos:
