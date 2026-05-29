Spectral diagnostics: periodogram, AR, Burg, and Capon
======================================================

Motivation
----------

Lattice and AR models are easier to understand when the documentation shows
spectra, not only coefficients.  The tutorial gallery therefore includes
visual diagnostics that compare nonparametric and model-based spectrum estimates
on controlled synthetic signals.

Periodogram
-----------

The periodogram estimates signal power directly from a windowed DFT:

.. math::

   \hat S_{per}(\omega)=\left|\sum_{n=0}^{N-1} w[n]x[n]e^{-j\omega n}\right|^2.

It is simple and robust as a baseline, but short records and window leakage can
make nearby peaks broad or hard to separate.

AR and Burg spectra
-------------------

An AR model writes

.. math::

   x[n]+a_1x[n-1]+\cdots+a_px[n-p]=e[n].

After estimating the denominator ``A(z)``, the all-pole spectral shape is

.. math::

   \hat S_{AR}(\omega) \propto \frac{1}{|\hat A(e^{j\omega})|^2}.

Levinson-Durbin estimates the AR denominator from autocorrelations.  Burg's
method estimates reflection coefficients from forward/backward prediction
errors.  Both can provide sharp spectral peaks, but both depend on model order.

Capon / MVDR spectrum
---------------------

Capon spectral estimation uses an inverse covariance matrix.  For steering
vector ``a(ω)`` and loaded covariance matrix ``R``, the spectrum is

.. math::

   \hat S_{Capon}(\omega)=\frac{1}{a(\omega)^H R^{-1} a(\omega)}.

It is a useful high-resolution diagnostic for nearby tones.  It is also more
sensitive to covariance estimation, aperture length, sample count, and diagonal
loading than a basic periodogram.

Tutorials
---------

The most useful pages in the generated tutorial gallery are:

* ``periodogram_vs_ar_spectrum.py``: periodogram versus Levinson/Burg AR spectra.
* ``capon_spectrum_demo.py``: Capon/MVDR compared with periodogram and AR.
* ``spectral_diagnostics_comparison.py``: side-by-side tuning of AR order and Capon aperture.

Build the rendered pages with figures and CSV downloads using:

.. code-block:: bash

   ./scripts/build_docs_with_results.sh
