Causality, data use, and signal roles
=====================================

This page fixes the vocabulary used throughout the examples.  The package
contains runtime filters, batch estimators, offline diagnostics, and finite-block
transforms.  They can all be useful, but they make different promises about
what data are available when an output or prediction is formed.

Causal one-step prediction
--------------------------

In the prediction examples, causal means that a prediction for the current sample
or vector is made before the current value is observed.  For a scalar signal,

.. math::

   \widehat y[n]
      = g\bigl(y[n-1], y[n-2], \ldots\bigr).

For a multichannel signal :math:`y[n]\in\mathbb{R}^c`, causal MIMO prediction
allows cross-channel history but still forbids future samples:

.. math::

   \widehat y[n]
      = g\bigl(y[n-1], y[n-2], \ldots\bigr),
      \qquad y[n-k]\in\mathbb{R}^c,\ k\ge 1.

Equivalently, a vector AR predictor has

.. math::

   \widehat y[n] = -\sum_{k=1}^p A_k y[n-k],
   \qquad A_k\in\mathbb{R}^{c\times c}.

The matrices :math:`A_k` may mix channels, so channel ``0`` at time ``n`` may
use channel ``2`` at time ``n-1``.  It may not use channel ``2`` at time
``n`` or any future time unless the task is explicitly noncausal/offline.

Causal filtering
----------------

For input-output filtering, causal means the output at time ``n`` uses only the
current input, previous inputs, and previous state:

.. math::

   y[n] = \sum_{i=0}^{q} b_i x[n-i]
          - \sum_{j=1}^{p} a_j y[n-j].

With the impulse-response convention

.. math::

   y[n] = \sum_{k\in\mathbb{Z}} h[k] x[n-k],

causality means :math:`h[k]=0` for :math:`k<0`, because negative ``k`` would
use future input samples :math:`x[n+|k|]`.

The scalar lattice filters, lattice-ladder filters, streaming block processor,
state-space simulator, online MIMO lattice predictor, and online matrix-lattice
all-pass runtime are causal in this sense.

Batch estimation versus causal use
----------------------------------

An estimator may use a full finite training record and still produce a model
that is later used causally.  This is the case for Burg, Levinson-Durbin, block
Levinson-Durbin, and finite-Hankel/model-reduction routines.

The distinction is:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Step
     - Data-use promise
   * - Coefficient estimation
     - May use a complete training record, covariance sequence, impulse response, or Markov tensor.
   * - Held-out prediction/filtering
     - Uses the fitted coefficients with no lookahead on the held-out stream.
   * - Frequency-response or Hankel diagnostics
     - Offline analysis of a finite object; useful for inspection, not a streaming runtime by itself.

In docs language, ``batch`` or ``offline`` describes the estimation/diagnostic
step.  ``Causal`` describes the runtime or prediction step.

Noncausal and finite-block transforms
-------------------------------------

Some examples intentionally use finite blocks or frequency-domain multiplication:

.. math::

   Y[\ell] = H(e^{j\omega_\ell}) X[\ell].

This is a circular/block operation unless embedded in an overlap-add or
state-space runtime.  It is useful for checking unitarity, adjoint identities,
perfect-reconstruction diagnostics, and compression of frequency-dependent
responses.  It should not be read as a sample-by-sample streaming filter unless
the page explicitly uses a streaming runtime such as
``OnlineMatrixLatticeAllPass``.

For all-pass systems there is also an important inverse distinction.  The
forward all-pass can be causal and stable.  Its time-domain impulse response has

.. math::

   y[n] = \sum_{k\ge 0} H_k x[n-k].

The finite-record adjoint used for reconstruction diagnostics is

.. math::

   x_{adj}[n] = \sum_{k\ge 0} H_k^H y[n+k].

That adjoint is time-domain but noncausal as an online inverse because it needs
future transformed samples.  The docs therefore separate ``causal forward
runtime`` from ``finite-record adjoint/perfect-reconstruction diagnostic``.

Transductive, inductive, and held-out language
----------------------------------------------

The examples avoid using future samples of the held-out test record when they
claim online prediction.  A useful vocabulary is:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Term
     - Meaning in these docs
   * - Online causal
     - Output or prediction is produced left-to-right with no future samples from the same stream.
   * - Inductive evaluation
     - Fit coefficients on a training record, then run causally on a separate held-out record.
   * - Transductive/offline evaluation
     - Use the whole record being evaluated to choose coefficients or a transform before reporting results on that same record.
   * - Block/circular
     - Process a complete finite block, usually through an FFT or dense matrix operator.

Transductive experiments can be valid diagnostics, especially for response
compression or finite-block unitary layers, but they are not the same claim as
online prediction on a held-out stream.

Single-signal prediction versus two-signal filtering
----------------------------------------------------

Prediction uses one observed signal.  At time ``n`` the model predicts the
current sample from its own past:

.. math::

   \widehat y[n] = g(y[n-1], y[n-2], \ldots),
   \qquad e[n]=y[n]-\widehat y[n].

Echo cancellation and system identification use two signals with different
physical roles.  A reference or far-end signal :math:`x[n]` drives an unknown
path, and the microphone or desired signal :math:`m[n]` contains the echo plus
other content:

.. math::

   \widehat e_{echo}[n] = h_\theta(x)[n],
   \qquad r[n] = m[n] - \widehat e_{echo}[n].

A causal echo-path filter may use current and past reference samples
:math:`x[n],x[n-1],\ldots`.  It should not use future reference samples or
future microphone samples.  In real double-talk conditions, near-end speech and
nonstationarity add control logic beyond the package's synthetic metric demos;
those examples are therefore presented as controlled DSP tests, not production
echo cancellers.

Package examples by data-use type
---------------------------------

The detailed MIMO verification map in :doc:`../algorithms/mimo_verification_map`
turns this vocabulary into concrete checks for the multichannel examples.


.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Example family
     - Data-use status
   * - Scalar lattice IIR, lattice-ladder, streaming block processing
     - Causal filtering of one input stream.
   * - Adaptive lattice/NLMS/RLS examples
     - Causal adaptive updates from the current input/desired pair after output and error are formed.
   * - Burg, Levinson, block Levinson, AR spectral estimation
     - Batch coefficient estimation; fitted AR predictors are causal once coefficients are fixed.
   * - ``MIMOLatticePredictor`` examples
     - Causal online vector prediction after a batch or precomputed coefficient fit.
   * - ``OnlineMatrixLatticeAllPass`` examples
     - Causal forward multichannel all-pass filtering.
   * - Paraunitary, ML-style unitary convolution, adjoint reconstruction demos
     - Forward analysis now uses streaming matrix-lattice runtimes where applicable; reconstruction checks use finite-record noncausal time-domain adjoints.
   * - Finite-Hankel, Nehari/AAK-style, and response-compression pages
     - Offline finite-dimensional analysis or model selection; resulting reduced models may be causal when implemented as state-space/IIR systems.
