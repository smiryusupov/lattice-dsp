AR estimation: autocorrelation, Levinson-Durbin, and Burg
=========================================================

Autoregressive model
--------------------

An AR(``p``) process is modeled as

.. math::

   x[n] + a_1 x[n-1] + \cdots + a_p x[n-p] = e[n],

where ``e[n]`` is a prediction error.  AR denominators are closely related to
reflection/PARCOR coefficients, so AR estimation is a natural companion to
lattice filtering.

Autocorrelation
---------------

``autocorrelation(x, order)`` computes the biased sample autocorrelation values
needed by the Yule-Walker equations:

.. math::

   r[k] = \frac{1}{N}\sum_{n=k}^{N-1} x[n]x[n-k].

Levinson-Durbin recursion
-------------------------

Levinson-Durbin solves the Toeplitz Yule-Walker system efficiently.  Each order
adds one reflection coefficient and updates the denominator coefficients and
prediction error power:

.. math::

   k_m = -\frac{r[m] + \sum_{i=1}^{m-1} a_i^{(m-1)} r[m-i]}{E_{m-1}},

.. math::

   E_m = E_{m-1}(1 - k_m^2).

Depending on sign convention, packages may report the negative of the
prediction-error reflection coefficient.  ``lattice-dsp`` uses one convention
consistently across ``levinson_durbin_*`` and denominator conversion helpers.

Burg recursion
--------------

The Burg method estimates AR parameters from forward and backward prediction
errors instead of first estimating a full autocorrelation sequence.  It is often
used for high-resolution spectral estimation from short records.  ``burg_reflection``
returns reflection coefficients and ``burg_denominator`` returns the corresponding
AR denominator.

The implementation uses compact forward/backward error vectors at each order so
stale boundary samples do not corrupt later reflection coefficients.

Batch estimation versus causal filtering
-----------------------------------------

Burg and Levinson-Durbin in this package are estimation algorithms, not runtime
streaming filters.  ``burg_reflection(x, order)`` uses the supplied finite
record ``x`` to compute reflection coefficients; ``levinson_durbin_*`` uses a
finite autocorrelation sequence.  In that sense, coefficient estimation is
batch/offline unless the caller maintains an online autocorrelation or sliding
window externally.

After estimation, the AR denominator represents a causal recursive model,

.. math::

   x[n] = -a_1 x[n-1] - \cdots - a_p x[n-p] + e[n],

and the corresponding IIR or prediction-error filter only needs past samples
and its current state.

Relevant APIs
-------------

* ``autocorrelation``
* ``levinson_durbin_reflection``
* ``levinson_durbin_denominator``
* ``levinson_durbin_error``
* ``burg_reflection``
* ``burg_denominator``

Examples
--------

* ``examples/burg_levinson_ar_tools.py``
* ``examples/ar_spectral_estimation.py``
* ``examples/adaptive_prediction_ar.py``

References
----------

See :doc:`../references` for Burg maximum-entropy spectral analysis, Durbin's
Toeplitz recursion, and adaptive-filtering books.
