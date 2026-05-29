Multichannel AR / block Levinson API
====================================

.. automodule:: lattice_dsp.multichannel_ar
   :members:
   :undoc-members:
   :show-inheritance:

Online MIMO predictor contract
------------------------------

``MIMOLatticePredictor`` is a causal vector predictor.  Coefficients may be
fitted offline with ``block_levinson_durbin``, but runtime prediction follows a
strict predict-before-update contract:

.. code-block:: python

   predictor = lattice_dsp.MIMOLatticePredictor.from_levinson(fit)

   y_hat = predictor.predict()   # uses only previous vectors
   error = predictor.update(y_t) # consumes the current vector afterward

For diagonal reflection matrices, the predictor decomposes into independent
one-channel/SISO predictors.  For full matrix reflections, previous samples from
one channel can help predict another channel.

.. autosummary::

   lattice_dsp.MIMOLatticePredictor
   lattice_dsp.causal_mimo_lattice_predict
   lattice_dsp.block_levinson_durbin

