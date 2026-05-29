Multichannel AR and block Levinson
==================================

This page describes the vector autoregressive (VAR) and block Levinson tools in
``lattice-dsp``.  They complement the matrix-lattice all-pass layer: both use
matrix-valued reflection ideas, but they solve different DSP problems.  The
MIMO verification map summarizes the public examples and tests that exercise
these distinctions; see :doc:`mimo_verification_map`.

Problem setting
---------------

A multichannel AR model of order ``p`` is written as

.. math::

   x[n] + A_1 x[n-1] + \cdots + A_p x[n-p] = e[n],

where ``x[n]`` is an ``m``-channel vector, each ``A_i`` is an ``m x m`` matrix,
and ``e[n]`` is the innovation/prediction error.

The package uses the autocorrelation convention

.. math::

   R[k] = E\{x[n]x[n-k]^H\}, \quad k \ge 0.

The block Yule-Walker equations are

.. math::

   \sum_{j=1}^{p} A_j R[i-j] = -R[i], \quad i=1,\ldots,p,

with ``R[-k] = R[k]^H``.  These equations form a block Toeplitz linear system.

Dense block-Toeplitz baseline
-----------------------------

``solve_block_yule_walker_direct`` builds the block Toeplitz matrix explicitly
and solves the full dense system.  This is useful as a correctness baseline and
for small models.

The dense solve has cubic scaling in the full problem size, roughly
``O((pm)^3)`` for order ``p`` and channel count ``m``.

Batch estimation versus causal VAR use
--------------------------------------

``multichannel_autocorrelation`` and ``block_levinson_durbin`` are batch
estimation routines.  They use a finite multichannel record, or the covariance
sequence derived from it, to estimate the matrices ``A_1, ..., A_p``.  That is
similar in spirit to scalar Burg/Levinson estimation: the estimator uses the
available history to identify coefficients.

The fitted VAR model itself is causal.  Once the matrices are known, prediction
or simulation only uses current/past samples and the current state/history,

.. math::

   x[n] = -\sum_{k=1}^{p} A_k x[n-k] + e[n].

Online MIMO lattice prediction
------------------------------

``MIMOLatticePredictor`` turns the forward/backward reflection matrices from
``block_levinson_durbin`` into a causal, stateful vector predictor.  The
estimation step is still batch, but the fitted lattice predictor can be used
one vector at a time.

For each stage ``m`` it keeps delayed backward-error vectors and applies

.. math::

   f_0[n] = b_0[n] = x[n],

.. math::

   f_m[n] = f_{m-1}[n] + K_m b_{m-1}[n-1],

.. math::

   b_m[n] = b_{m-1}[n-1] + L_m f_{m-1}[n].

The one-step prediction is available before the current vector is observed:

.. math::

   \hat{x}[n] = -f_p[n]\big|_{x[n]=0}.

After ``update(x[n])`` the returned forward error is

.. math::

   e[n] = f_p[n] = x[n] - \hat{x}[n].

This is the online/causal MIMO lattice path in the public API.  It is separate
from the matrix all-pass FFT/block examples, which remain frequency-domain
diagnostics in this release.

.. code-block:: python

   r = ld.multichannel_autocorrelation(x_train, order=4)
   fit = ld.block_levinson_durbin(r, order=4)
   predictor = ld.MIMOLatticePredictor.from_levinson(fit)

   y_hat = predictor.predict()   # uses only previous vectors
   error = predictor.update(y_t) # consumes the current vector

Block Levinson-Durbin recursion
-------------------------------

``block_levinson_durbin`` implements the order-recursive forward/backward block
Levinson-Durbin recursion, also known in the multichannel prediction literature
through Whittle and Wiggins-Robinson/LWR-style algorithms.

At each order the recursion computes a forward matrix reflection coefficient
``K_f`` and a backward matrix reflection coefficient ``K_b`` from the residual
forward/backward covariance.  The forward AR coefficients are then updated from
the previous-order forward and backward predictors.

The scalar Levinson-Durbin algorithm returns PARCOR/reflection coefficients.
The block version returns **matrix reflection coefficients**.  Their largest
singular values are useful diagnostics: well-behaved stable models usually keep
these values below one, analogous to scalar reflection coefficients staying
inside the unit circle.


Diagonal and coupled sanity checks
----------------------------------

The examples include two complementary online checks.

* :doc:`../examples/generated/mimo_diagonal_equals_independent_siso` verifies
  that diagonal matrix reflections reduce exactly to independent one-channel
  predictors.
* :doc:`../examples/generated/online_coupled_mimo_vs_siso` fits a coupled vector
  AR process and compares full MIMO prediction with diagonal and independent
  SISO baselines.  The key diagnostic is not only residual RMS, but also the
  off-diagonal residual correlation left after prediction.

These checks are useful because they separate implementation sanity from the
actual reason to use MIMO: cross-channel history.

Relationship to matrix lattice all-pass filters
-----------------------------------------------

The multichannel AR/block-Levinson layer and the matrix all-pass lattice layer
are related but not identical.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Tool
     - Main purpose
   * - Block Levinson / vector AR
     - Estimate multichannel predictors from block Toeplitz covariance.
   * - Matrix lattice all-pass
     - Build compact unitary/paraunitary MIMO transfer functions.

This distinction is important for positioning.  ``lattice-dsp`` is not trying to
force every MIMO problem into one algorithm.  It provides a coherent family of
scalar and matrix lattice tools for stable recursive and multichannel DSP.

Important API symbols
---------------------

* ``multichannel_autocorrelation``
* ``block_toeplitz_from_autocorrelation``
* ``solve_block_yule_walker_direct``
* ``block_levinson_durbin``
* ``MIMOLatticePredictor``
* ``causal_mimo_lattice_predict``
* ``multichannel_prediction_error``
* ``matrix_ar_frequency_response``
* ``companion_spectral_radius``

Example
-------

.. code-block:: python

   import numpy as np
   import lattice_dsp as ld

   x = np.random.default_rng(0).normal(size=(8192, 3))
   r = ld.multichannel_autocorrelation(x, order=4)

   direct = ld.solve_block_yule_walker_direct(r, order=4)
   recursive = ld.block_levinson_durbin(r, order=4)

   print(np.linalg.norm(direct.coefficients - recursive.coefficients))
   print(recursive.reflection_spectral_norms)

Examples and benchmark
----------------------

Run from the repository root:

.. code-block:: bash

   python examples/multichannel_levinson_ar.py
   python examples/causal_mimo_lattice_prediction.py
   python examples/matrix_ar_spectral_estimation.py
   python examples/mimo_lattice_vs_block_levinson.py
   python benchmarks/block_levinson_benchmark.py --channels 4 --order 12

References
----------

See :doc:`../references` for classical references on scalar Levinson-Durbin,
multichannel Levinson/Whittle/LWR recursions, matrix lattice filters, and
paraunitary systems.
