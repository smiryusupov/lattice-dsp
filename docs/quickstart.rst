Quickstart
==========

This quickstart shows the core idea of ``lattice-dsp``: represent recursive
filters with reflection/PARCOR coefficients and lattice-ladder structures rather
than treating IIR denominator coefficients as unconstrained parameters.

For a full guided walkthrough, build the generated tutorial gallery:

.. code-block:: bash

   ./scripts/build_docs_with_results.sh
   xdg-open docs/_build/html/examples/index.html

Reflection coefficients to a stable IIR denominator
---------------------------------------------------

Reflection coefficients satisfy ``abs(k_i) < 1``.  For scalar all-pole
denominators, this gives a stability-aware parameterization: bounded reflection
coefficients map to stable denominators.

.. code-block:: python

   import numpy as np
   import lattice_dsp as ld

   k = np.array([0.7, -0.4, 0.25])
   a = ld.reflection_to_denominator(k)
   restored = ld.denominator_to_reflection(a)

   print("denominator", a)
   print("restored", restored)

Lattice-ladder realization
--------------------------

A lattice-ladder filter uses reflection coefficients for the recursive part and
ladder taps for the numerator/output combination.  This is the scalar IIR path
that most directly motivates the package.

.. code-block:: python

   filt = ld.LatticeLadderIIR([0.35, -0.2, 0.1], [0.1, -0.05, 0.2, 0.8])
   x = np.random.default_rng(0).standard_normal(1024)
   y = filt.process(x)

Adaptive recursive filtering
----------------------------

The adaptive examples are intended to demonstrate stable recursive
identification.  They are not generic black-box optimizers; the point is to
update a recursive model while controlling the denominator through lattice
coordinates.

.. code-block:: python

   model = ld.LatticeLadderRLS(order=3, forgetting_factor=0.99, delta=100.0)
   x = np.random.default_rng(0).standard_normal(4096)
   d = np.convolve(x, [0.25, -0.15, 0.65], mode="full")[: len(x)]
   y, e = model.process(x, d)
   print("tail MSE", np.mean(e[-512:] ** 2))

Matrix/MIMO lattice example
---------------------------

The matrix lattice utilities explore multichannel all-pass and paraunitary
behavior.  They are useful for experiments in compact MIMO responses,
energy-preserving transforms, multichannel AR diagnostics, and ML-inspired
unitary layers.

.. code-block:: python

   rng = np.random.default_rng(0)
   K = ld.contractive_matrix_from_raw(
       0.2 * (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))),
       margin=1e-6,
   )
   R = ld.unitary_polar_factor(rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))
   filt = ld.MatrixLatticeAllPass([K], R)
   H = filt.frequency_response(np.linspace(0, np.pi, 128))
   y = filt.to_online_filter().process(np.ones((64, 3)))

   # H[w]^H H[w] should be close to identity for every frequency.
   I = np.eye(4)
   err = max(np.linalg.norm(h.conj().T @ h - I) for h in H)
   print(err)


Causal MIMO lattice prediction
------------------------------

The multichannel AR layer can also run as a sample-by-sample lattice predictor
after its matrix reflection coefficients have been fitted.

.. code-block:: python

   x_train = np.random.default_rng(1).standard_normal((4096, 3))
   r = ld.multichannel_autocorrelation(x_train, order=3)
   fit = ld.block_levinson_durbin(r, order=3)
   pred = ld.MIMOLatticePredictor.from_levinson(fit)

   y_hat = pred.predict()       # before observing y_t
   err = pred.update(x_train[0]) # after observing y_t

Flagship robustness tutorial
----------------------------

After installing the docs/examples extras, build the generated tutorials and open
the :doc:`LMS/H∞ robustness tutorial <examples/generated/hinf_lms_reproduction>`:

.. code-block:: bash

   ./scripts/build_docs_with_results.sh
   xdg-open docs/_build/html/examples/generated/hinf_lms_reproduction.html

The rendered page is also available from the docs navigation at
:doc:`Examples tutorials / LMS as a minimax robust filter <examples/generated/hinf_lms_reproduction>`.
This tutorial explains why LMS can be interpreted as a minimax robust filter,
not only as an approximate least-squares recursion.
