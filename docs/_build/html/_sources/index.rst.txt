lattice-dsp documentation
=========================

``lattice-dsp`` is a focused C++/Python toolkit for efficient stable IIR
lattice filters, lattice-ladder realizations, adaptive recursive DSP, AR
spectral modeling, and matrix/MIMO lattice experiments.

The package is intentionally narrower than a general DSP library.  Its center is
reflection/PARCOR and lattice parameterizations: coordinates where scalar IIR
stability can be controlled directly and where matrix-valued all-pass,
paraunitary, and multichannel AR examples can be studied in a reproducible
Python workflow.

The niche is the combination.  ``lattice-dsp`` puts stable SISO IIR lattice
filters, finite model-reduction diagnostics, MIMO block-Hankel reduction,
compiled MIMO state-space simulation, matrix-lattice all-pass scaffolds, and
finite tangential Schur/Pick/J-inner diagnostics in one C++/Python package.  The MIMO model-reduction layer is a particularly
specialized part of the package, so the docs use explicit scope and validation
language rather than broad solver claims.

What to use it for
------------------

* Stable scalar IIR filters parameterized by reflection/PARCOR coefficients.
* Lattice-ladder IIR realizations with numerator/ladder conversions.
* Adaptive NLMS and RLS experiments for recursive filters.
* H∞/minimax LMS robustness tutorial inspired by Hassibi, Sayed, and Kailath.
* Burg, Levinson-Durbin, AR, Capon, and periodogram diagnostics.
* A lattice DSP concept map connecting Schur/Szegő theory, all-pass completions, Hankel/Padé approximation, and matrix/MIMO lattice forms.
* Package-positioning and SISO/MIMO theory maps: Hankel operators, Nehari, AAK, reachability, and observability.
* Dependency-free interoperability recipes for Pyroomacoustics-style RIRs, eSpeak/eSpeak NG WAV sources, and optional audio loaders.
* Multichannel/block Levinson AR estimation, dense block-Toeplitz baselines, and causal online MIMO lattice prediction.
* Matrix/MIMO lattice all-pass and paraunitary response experiments.
* Finite right-tangential Schur/Pick diagnostics and elementary J-inner Potapov factors.
* OpenMP-backed batch processing for many independent filters/signals.
* Tutorial pages that run examples and embed output, figures, data, and source.

Online versus offline language
------------------------------

The runtime filters are documented separately from coefficient-estimation and
diagnostic routines.  Scalar lattice IIR filters, the streaming block processor,
``MIMOLatticePredictor``, ``mimo_state_space_process_batch``, and
``OnlineMatrixLatticeAllPass`` are causal runtime objects.  Burg,
Levinson-Durbin, block Levinson, finite-Hankel reduction, and response
compression are batch/offline steps that may use a full training record or
finite response object before a causal model is run.

The online MIMO lattice examples use the contract
``prediction = predictor.predict()`` before ``predictor.update(y_t)``.  The
package also keeps finite-block paraunitary and ML-style unitary demos, but those
are labelled as block/circular diagnostics when they use an adjoint inverse or a
whole FFT block.  See :doc:`theory/causality_and_data_use` for the terminology
used by the examples.

Non-goals
---------

``lattice-dsp`` is not a production acoustic echo canceller, not a replacement
for SciPy, and not a complete room-acoustics or wireless-communications
framework.  Synthetic ERLE, equalization, and MIMO examples are included as
controlled DSP demonstrations and diagnostics.

Quick navigation
----------------

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart
   learning_path
   interoperability


.. toctree::
   :maxdepth: 2
   :caption: Theory guide

   Theory overview <theory/index>

.. toctree::
   :maxdepth: 2
   :caption: Algorithms

   Algorithms overview <algorithms/index>

.. toctree::
   :maxdepth: 2
   :caption: Examples and benchmarks

   examples/index
   benchmarks/index

.. toctree::
   :maxdepth: 2
   :caption: API reference

   API overview <api/index>

.. toctree::
   :maxdepth: 1
   :caption: Project

   project_status
   release_notes_0_1
   development
   references

Minimal example
---------------

.. code-block:: python

   import numpy as np
   import lattice_dsp as ld

   # Stable second-order IIR denominator through reflection coefficients.
   k = np.array([0.7, -0.4])
   a = ld.reflection_to_denominator(k)
   print(a)

   # Matrix-valued all-pass / paraunitary frequency response.
   rng = np.random.default_rng(0)
   K = ld.contractive_matrix_from_raw(0.2 * rng.standard_normal((3, 3)))
   R = ld.unitary_polar_factor(rng.standard_normal((3, 3)))
   filt = ld.MatrixLatticeAllPass([K], R)
   H = filt.frequency_response(np.linspace(0, np.pi, 64))
   y = filt.to_online_filter().process(np.ones((32, 3)))
   print(H.shape, y.shape)
