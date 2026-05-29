Release notes for the 0.1 public alpha
======================================

This page summarizes the first public-alpha ``lattice-dsp`` 0.1 validation snapshot.
The numbers below came from one local Linux/OpenMP workstation run and are
presented as reproducible validation examples, not universal performance
guarantees.

Scope and niche
---------------

The 0.1 public alpha emphasizes a rare combination for a Python DSP package: stable SISO
IIR lattice/lattice-ladder filtering, finite-Hankel and finite-section
Nehari/AAK-style model-reduction diagnostics, MIMO block-Hankel reduction,
compiled MIMO state-space simulation, and matrix-lattice all-pass scaffolds.

The MIMO features are presented as finite-dimensional baselines and bridge
diagnostics.  They are intentionally documented separately from exact
matrix-valued AAK/Nehari or arbitrary MIMO matrix-lattice realization claims,
which are outside the 0.1 public-alpha scope.

Validation snapshot
-------------------

The current tree is validated with the full pytest suite plus a release-trust
layer that checks the public claims directly: reflection-bounded stability,
adaptive-IIR stability failure modes, finite SISO/MIMO model-reduction
diagnostics, small-scale smoke runs of the flagship long-signal tutorials, and
public overclaim wording guards.

.. code-block:: bash

   pytest -q
   pytest -q tests/test_release_trust_claims.py

The public API audit reported 18 classes and 64 functions when compatibility
aliases are shown.  With deprecated aliases hidden, the public function count is
62.  The two hidden aliases are:

.. code-block:: text

   finite_hankel_aak_reduce_iir      -> finite_hankel_reduce_iir
   finite_hankel_aak_reduce_impulse  -> finite_hankel_reduce_impulse

They remain available for compatibility, but new code should use the
``finite_hankel_reduce_*`` names.

Validated capabilities
----------------------

.. list-table::
   :header-rows: 1
   :widths: 28 47 25

   * - Area
     - What is validated
     - Status
   * - Scalar lattice/IIR filters
     - reflection/PARCOR conversion, lattice and lattice-ladder filtering,
       stable adaptive examples
     - implemented
   * - Spectral diagnostics
     - Levinson, Burg, AR spectra, periodogram comparisons, and Capon/MVDR
       diagnostic examples
     - implemented
   * - Finite-Hankel SISO reduction
     - finite-section Ho--Kalman-style baseline for stable IIR reduction
     - implemented
   * - Finite-section SISO Nehari/AAK workflow
     - tail approximation, rational candidates, rank selection, Schmidt-pair
       certificate, exact rational-tail validation, noisy-tail validation, and
       IIR reduction demo
     - implemented as finite-section baseline
   * - MIMO block-Hankel reduction
     - diagonal sanity check, coupled state-space tutorial, finite block-Hankel
       reduction, and compiled batched state-space processing
     - implemented as MIMO baseline
   * - Full infinite-dimensional SISO AAK/Nehari solver
     - exact operator-level solver beyond finite sections
     - outside 0.1 scope
   * - Matrix AAK/Nehari solver
     - deeper MIMO/matrix extension
     - outside 0.1 scope

Headline SISO result
--------------------

The finite-section AAK/Nehari IIR tutorial reduces an eighth-order stable IIR to
an automatically selected order-3 candidate.  A representative run reported:

.. code-block:: text

   selected rank: 3
   selected accepted: True
   selected pole radius: 0.9074
   relative impulse error: 3.863e-03
   batch output SNR: 48.27 dB
   batch output rel MSE: 1.488e-05
   max magnitude error: 0.039 dB
   filter speedup: 1.72x

The corresponding finite-section IIR benchmark showed the intended division of
roles: finite-Hankel reduction is currently the fastest practical baseline, while
the finite-section AAK/Nehari candidate workflow is valuable for rank selection,
certificates, rational diagnostics, and selected high-quality reductions.  It is
not always faster end-to-end.

Headline MIMO result
--------------------

The coupled MIMO tutorial validates reduction quality on a dense 3-input,
3-output state-space system.  A representative run reported:

.. code-block:: text

   full state order: 12
   full state spectral radius: 0.8800
   order=6: stable=True, radius=0.8562, retained=0.997184,
            markov_error=3.630e-03, output_snr=24.45 dB
   order=8: stable=True, radius=0.8809, retained=0.999900,
            markov_error=1.589e-04, output_snr=38.02 dB

The compiled MIMO benchmark separates processing speedup from preprocessing
cost.  With 3 inputs, 3 outputs, batch size 8, 6,000 samples per batch,
``reuse_count=50``, and one compiled worker thread, a representative run
reported:

.. list-table::
   :header-rows: 1
   :widths: 10 10 16 16 16 16 16

   * - full
     - red
     - process speedup
     - one-shot speedup
     - reuse speedup
     - SNR
     - Markov error
   * - 16
     - 4
     - 4.54x
     - 0.12x
     - 2.62x
     - 16.15 dB
     - 2.411e-02
   * - 16
     - 6
     - 4.69x
     - 0.14x
     - 2.85x
     - 26.14 dB
     - 2.423e-03
   * - 16
     - 8
     - 3.86x
     - 0.17x
     - 2.69x
     - 32.66 dB
     - 5.397e-04

This is the preferred interpretation of the MIMO runtime story:

* ``proc_x`` measures repeated state-space processing only;
* ``one_x`` includes one reduction and one processing batch, so it can stay
  below one because reduction is a preprocessing cost;
* ``reuse_x`` amortizes the reduced model over repeated batches and is the most
  relevant number for reusable reduced models.

Recommended release checks
--------------------------

Run these before tagging or publishing a release candidate:

.. code-block:: bash

   python -m pip install -e '.[dev,examples,benchmark,docs]'
   PYTHONPATH="$PWD" pytest -q
   ./scripts/build_docs_with_results.sh
   PYTHONPATH="$PWD" python tools/audit_public_api.py
   PYTHONPATH="$PWD" python tools/audit_public_api.py --hide-deprecated

Representative benchmark commands
---------------------------------

SISO finite-Hankel versus finite-section AAK/Nehari IIR reduction:

.. code-block:: bash

   python benchmarks/finite_aak_iir_reduction_speedup.py \
     --full-orders 8 16 32 \
     --target-orders 3 4 6 8 12 \
     --channels 32 \
     --samples 30000 \
     --repeats 3 \
     --output reports/finite-aak-iir-reduction-speedup.json

Coupled MIMO block-Hankel reduction with compiled state-space processing:

.. code-block:: bash

   python benchmarks/mimo_hankel_reduction_speedup.py \
     --full-orders 8 16 \
     --reduced-orders 2 4 6 8 \
     --inputs 3 \
     --outputs 3 \
     --batch 8 \
     --samples 6000 \
     --repeats 2 \
     --reuse-count 50 \
     --n-threads 1 \
     --output reports/mimo-hankel-reduction-speedup.json

Public scope and validation boundary
------------------------------------

The 0.1 public alpha has a coherent baseline ladder:

.. code-block:: text

   scalar stable lattice/IIR filters
   -> finite-Hankel SISO reduction
   -> finite-section SISO Nehari/AAK diagnostics and IIR candidate reduction
   -> MIMO diagonal sanity checks
   -> coupled MIMO block-Hankel reduction with compiled state-space processing

The public 0.1 scope is the validated baseline ladder above.  Larger MIMO runtime studies, stronger noisy/non-rational SISO AAK/Nehari validations, operator-level SISO solvers, and matrix AAK/Nehari solvers are outside this release scope and are documented separately from the supported finite-section APIs.
