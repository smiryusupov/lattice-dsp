Filtering relationships and signal-processing vocabulary
========================================================

This page collects basic signal-processing terminology used by the examples.
It is intentionally concrete: the goal is to make equalization, echo
cancellation, prediction, adaptivity, recursivity, and phase terminology mean
specific mathematical objects before readers encounter the lattice and MIMO
algorithms.

For data-use language such as causal, batch/offline, transductive, and
finite-block/circular, see :doc:`causality_and_data_use`.  This page focuses on
the physical roles of the signals once that data-use convention is fixed.

Three filtering relationships
-----------------------------

Many adaptive-filter examples have the same algebraic core but different signal
roles.  A filter produces

.. math::

   y[n] = \sum_{i=0}^{q} b_i x[n-i]
          - \sum_{j=1}^{p} a_j y[n-j]

and an error signal compares that output with a desired signal ``d[n]``:

.. math::

   e[n] = d[n] - y[n].

The meaning of ``x``, ``d``, ``y``, and ``e`` changes by task.

.. list-table::
   :header-rows: 1
   :widths: 20 23 23 34

   * - Relationship
     - Input to adaptive filter
     - Desired signal / error
     - What the filter is learning
   * - Equalization
     - Received distorted signal ``r[n]``
     - Known training symbol or decision ``s[n]``; ``e[n]=s[n]-y[n]``
     - An inverse or partial inverse of a channel so that ``y[n]`` resembles the transmitted signal.
   * - Echo cancellation
     - Far-end/reference signal ``x[n]``
     - Microphone signal ``m[n]``; residual ``e[n]=m[n]-\hat e_{echo}[n]``
     - The echo path from loudspeaker/reference to microphone, so its estimate can be subtracted.
   * - Prediction
     - Past samples ``[x[n-1],x[n-2],\ldots]``
     - Current sample ``x[n]``; innovation ``e[n]=x[n]-\hat x[n]``
     - A model of temporal structure; the residual is the unpredictable innovation.

The same update law can therefore appear in different applications.  What
changes is the physical interpretation of the regressor and the error.

Single-signal prediction and two-signal echo cancellation
---------------------------------------------------------

The prediction examples are single-signal models.  The signal being predicted is
also the signal whose past is used as the regressor:

.. math::

   \widehat x[n] = g(x[n-1],x[n-2],\ldots),
   \qquad e[n]=x[n]-\widehat x[n].

A causal predictor must form :math:`\widehat x[n]` before observing
:math:`x[n]`.  Multichannel prediction changes the scalar sample into a vector,
but the data-use rule is the same: previous vectors may be used and future
vectors may not.

Echo cancellation and system identification use two signals.  The adaptive
filter receives a reference/far-end signal :math:`x[n]`; the desired or
microphone signal :math:`m[n]` is compared with the estimated path output:

.. math::

   \widehat d[n] = h_\theta(x)[n],
   \qquad r[n] = m[n] - \widehat d[n].

In echo-cancellation language, :math:`x[n]` is the loudspeaker/far-end reference,
:math:`\widehat d[n]` is the estimated echo, :math:`m[n]` is the microphone
mixture, and :math:`r[n]` is the residual sent downstream.  The synthetic ERLE
examples use this two-signal metric structure, but they do not implement the
control logic, delay estimation, nonlinearities, double-talk handling, or product
engineering expected from a production acoustic echo canceller.


Adaptivity
----------

A fixed filter has parameters that do not change with time.  An adaptive filter
has a parameter vector or state-dependent parameterization ``\theta_n`` that is
updated from streaming data:

.. math::

   y[n] = f_{\theta_n}(x_n), \qquad
   e[n] = d[n] - y[n], \qquad
   \theta_{n+1} = \theta_n + \Delta(\theta_n, x_n, d[n]).

For FIR LMS, for example,

.. math::

   w_{n+1} = w_n + \mu x_n e[n].

For normalized LMS,

.. math::

   w_{n+1} = w_n + \frac{\mu}{\epsilon + \|x_n\|_2^2} x_n e[n].

Even in the FIR case, the learning rate ``μ`` is a real design parameter rather
than a formality.  If it is too small, convergence is slow; if it is too large,
the recursion can have large misadjustment or diverge.  The safe range depends
on the signal power and input covariance, so practical step-size selection is
often a tuning problem rather than a one-line formula.

For lattice IIR adaptation, the parameter vector includes numerator/ladder
parameters and reflection coefficients.  The important distinction is that
adaptive recursive filters keep the LMS step-size problem and add a denominator
stability problem.  Reflection coefficients give a structural guard: in the
scalar all-pole case, keeping ``|k_i| < 1`` keeps the recursive denominator
stable.

Recursivity
-----------

A recursive filter feeds previous outputs or internal states back into the
current computation.  A direct-form IIR model is recursive because

.. math::

   y[n] = \sum_{i=0}^{q} b_i x[n-i]
          - \sum_{j=1}^{p} a_j y[n-j].

Equivalently, a state-space model is recursive because

.. math::

   x_{s}[n+1] = A x_{s}[n] + B u[n], \qquad
   y[n] = C x_{s}[n] + D u[n].

The pole locations of ``A(z)`` or the eigenvalues of ``A`` control stability.
This is why adaptive IIR filtering is more delicate than adaptive FIR filtering:
changing denominator or state-recursion parameters changes the stability of the
algorithm itself.

Minimum phase and maximum phase
-------------------------------

For a causal discrete-time transfer function

.. math::

   H(z)=\frac{B(z)}{A(z)},

poles determine stability and zeros determine phase/invertibility properties.
A stable causal scalar system is **minimum phase** when its zeros are also inside
the unit disk.  In that case the inverse filter is causal and stable, up to
delay and gain conventions.  Minimum-phase systems concentrate energy early in
the impulse response and are the easiest case for stable equalization.

A **maximum-phase** scalar system has zeros outside the unit disk.  Its stable
inverse would have poles outside the unit disk if implemented causally without
additional delay or noncausal processing.  Equalizing such a channel is therefore
numerically and structurally harder.

A useful scalar split is

.. math::

   H(z) = H_{\mathrm{outer}}(z) H_{\mathrm{inner}}(z).

The outer factor is the minimum-phase part that carries the invertible amplitude
information.  The inner/all-pass factor has unit magnitude on the unit circle
and carries phase or delay-like structure.  In lattice language, all-pass and
lossless completions are natural companions to this inner/outer viewpoint.

Long memory and speed
---------------------

The speed motivation for IIR filters is easiest to see from impulse-response
length.  A long FIR path stores many taps and computes

.. math::

   y[n] = \sum_{m=0}^{L-1} h[m] x[n-m].

For large ``L`` and millions of samples, this can be expensive even when block
or FFT convolution is used.  A recursive model can represent some long-memory
responses with far fewer parameters.  For example, the exponential tail

.. math::

   h[m] = (1-r) r^m, \qquad 0 < r < 1,

has the one-state recursion

.. math::

   y[n] = (1-r) x[n] + r y[n-1].

That is the numerical attraction of stable IIR and model-reduction workflows:
when a long impulse response has compact recursive structure, the implementation
can process very long signals using small state rather than a long tap vector.
The package includes a million-sample throughput tutorial that demonstrates this
point for an acoustic-like decay while keeping the benchmark local and
reproducible.

Why this matters for this package
---------------------------------

``lattice-dsp`` focuses on stable recursive coordinates.  The practical message
is:

* recursive filters are powerful because they represent long memory with few
  parameters;
* adaptive recursive filters are risky unless stability is parameterized or
  monitored;
* reflection coefficients provide a stable scalar IIR coordinate system;
* finite Hankel/model-reduction tools help look for compact recursive structure
  in long impulse-response or Markov-parameter data;
* MIMO and matrix cases replace scalar magnitudes by contraction and block-Hankel
  diagnostics;
* equalization, echo cancellation, and prediction use similar mathematics but
  different signal roles and validation metrics.
