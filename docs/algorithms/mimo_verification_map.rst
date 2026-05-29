MIMO verification map
=====================

The multichannel examples are intended to make specific, testable claims.  This
page collects those claims in one place so that the MIMO material can be read as
a verification story rather than as unrelated demonstrations.

The main distinction is between **causal forward runtimes** and
**finite-record diagnostics**.  A causal runtime produces an output or prediction
left-to-right without future samples from the stream being processed.  A
finite-record diagnostic may use a whole impulse response, covariance tensor,
frequency grid, or transformed record to check a property.  Both are useful, but
they should not be confused.

Summary table
-------------

.. list-table::
   :header-rows: 1
   :widths: 26 28 24 22

   * - Claim
     - Tutorial
     - Main diagnostic
     - Data-use status
   * - Diagonal MIMO reduces to independent SISO systems.
     - :doc:`../examples/generated/mimo_diagonal_equals_independent_siso`
     - Maximum output and online-prediction differences are at roundoff.
     - Causal runtime after fixed coefficients.
   * - Online MIMO lattice prediction uses no current/future target vector.
     - :doc:`../examples/generated/causal_mimo_lattice_prediction`
     - ``predict()`` is called before ``update(y_n)``; residuals match the fitted VAR model.
     - Batch fit, causal held-out prediction.
   * - Coupled MIMO prediction captures cross-channel history that SISO misses.
     - :doc:`../examples/generated/online_coupled_mimo_vs_siso`
     - Full-MIMO residual RMS and off-diagonal residual correlation are lower than SISO baselines.
     - Batch fit, causal held-out prediction.
   * - Block Levinson matches a direct block-Toeplitz solve.
     - :doc:`../examples/generated/multichannel_levinson_ar`
     - Coefficient difference, reflection norms, and residual covariance.
     - Batch coefficient estimation; fitted VAR is causal.
   * - Matrix AR spectra separate auto- and cross-channel behavior.
     - :doc:`../examples/generated/matrix_ar_spectral_estimation`
     - Spectral-density matrix, companion radius, and reflection norms.
     - Offline frequency-grid diagnostic of a fitted causal model.
   * - Matrix lattice all-pass runtime matches the frequency response.
     - :doc:`../examples/generated/matrix_lattice_allpass`
     - Impulse-response FFT versus frequency evaluator; singular values near one.
     - Causal forward runtime plus offline response check.
   * - Coupled matrix-lattice filtering is streaming and channel-mixing.
     - :doc:`../examples/generated/coupled_mimo_lattice_filter`
     - Off-diagonal Markov energy and streaming-vs-impulse agreement.
     - Causal forward runtime; finite adjoint is noncausal.
   * - Audio decorrelation changes covariance while preserving energy.
     - :doc:`../examples/generated/multichannel_audio_decorrelator`
     - Off-diagonal covariance/correlation reduction and total-energy ratio.
     - Causal forward runtime.
   * - Finite block-Hankel MIMO reduction behaves on slow, high-dimensional, and ill-conditioned Markov data.
     - :doc:`../examples/generated/mimo_model_reduction_stress_cases`
     - Relative Markov/H2 error, block-Hankel singular values, tangential sample residuals, and Pick eigenvalues.
     - Offline finite-data reduction and interpolation diagnostics.
   * - Matrix-lattice compression represents a dense unitary grid compactly.
     - :doc:`../examples/generated/matrix_unitary_response_compression`
     - Storage ratio, singular values, and approximation error.
     - Offline grid diagnostic; compact object has causal forward runtime.
   * - Paraunitary-style analysis is streaming; synthesis check is finite-record.
     - :doc:`../examples/generated/paraunitary_filter_bank_demo`
     - Energy preservation, singular values, and finite-adjoint reconstruction error.
     - Causal forward analysis; noncausal/transductive adjoint diagnostic.
   * - ML-style unitary convolution preserves norms in a streaming forward pass.
     - :doc:`../examples/generated/ml_unitary_convolution_demo`
     - Input/output norms with tail padding and finite-adjoint error.
     - Causal forward runtime; noncausal finite adjoint diagnostic.

Runtime categories
------------------

The examples use three recurring data-use categories.

Causal online prediction
   A predictor forms :math:`\widehat y[n]` before seeing :math:`y[n]`.  The
   online MIMO lattice predictor follows the explicit sequence

   .. code-block:: python

      y_hat = predictor.predict()   # uses stored past state only
      error = predictor.update(y_n) # consumes the current vector afterward

   The fitted reflections may come from a batch training record, but held-out
   prediction is left-to-right.

Causal forward filtering
   A filter computes :math:`y[n]` from the current input, previous inputs, and
   stored state.  The online matrix-lattice all-pass runtime realizes

   .. math::

      y[n] = \sum_{k\ge 0} H_k x[n-k].

   This is the runtime used by the streaming matrix-lattice and audio
   decorrelation examples.

Finite-record adjoint diagnostics
   Perfect-reconstruction and adjoint checks use the finite-record adjoint

   .. math::

      x_{adj}[n] = \sum_{k\ge 0} H_k^H y[n+k].

   This is time-domain, but it is noncausal as an online inverse because it uses
   future transformed samples.  The paraunitary and ML-style examples therefore
   describe the forward analysis path and the reconstruction diagnostic
   separately.

Why the diagonal and coupled checks both matter
-----------------------------------------------

The diagonal check prevents accidental overclaiming.  If every matrix is
diagonal, MIMO should reduce to independent SISO systems exactly.  The coupled
check then demonstrates the reason to keep full matrix coefficients: when a
channel depends on another channel's past, independent SISO predictors cannot
remove that cross-channel structure.  The package tests both cases because they
answer different questions:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Check
     - Meaning
   * - Diagonal MIMO equals SISO
     - The implementation respects the scalar limit and does not introduce artificial coupling.
   * - Coupled MIMO beats SISO
     - Full matrix reflections/coefficient matrices use cross-channel history that independent predictors ignore.

Connection to tests
-------------------

The test suite mirrors these public claims.  In particular, it checks
sample-by-sample equivalence between diagonal online MIMO predictors and stacked
one-channel predictors, prefix causality for online all-pass filtering, agreement
between online all-pass impulse responses and frequency responses, and a coupled
VAR case where full MIMO prediction improves on independent SISO baselines.

Tangential Schur/Pick baseline for matrix-lattice claims
--------------------------------------------------------

The MIMO lattice pages also link to finite tangential-Schur diagnostics.  These
are not streaming runtimes; they are offline interpolation certificates and
J-inner checks that make the matrix-valued Schur/Potapov background inspectable.

.. list-table::
   :header-rows: 1
   :widths: 30 30 25 15

   * - Claim
     - Example
     - Diagnostic
     - Data-use status
   * - Definite right-tangential data satisfy the Pick PSD certificate.
     - :doc:`../examples/generated/tangential_schur_pick_jinner`
     - Pick eigenvalues and constant interpolation residual.
     - Offline finite-data diagnostic.
   * - Elementary Potapov factors are J-inner on the unit circle.
     - :doc:`../examples/generated/tangential_schur_pick_jinner`
     - :math:`\max_\omega\|\Theta(e^{j\omega})^H J\Theta(e^{j\omega})-J\|_2` and graph-vector annihilation.
     - Offline J-inner diagnostic.
   * - Diagonal MIMO tangential data reduce to scalar Pick problems.
     - :doc:`../examples/generated/diagonal_tangential_schur_equals_scalar`
     - Full Pick matrix equals a scalar block-diagonal Pick matrix.
     - Offline reduction-to-scalar sanity check.
   * - MIMO model-reduction stress data can be read through sampled Schur/Pick certificates.
     - :doc:`../examples/generated/mimo_model_reduction_stress_cases`
     - Scale :math:`\gamma` until sampled right-tangential Pick matrix is PSD; compare H2/Markov error, finite-Hankel spectral-norm tail error, timing/backend labels, and reduced-model tangential residuals.
     - Offline interpolation diagnostic, not a complete Schur reduction solver.

