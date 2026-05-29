Benchmark tutorials
===================

These pages are generated from scripts in ``benchmarks/``.  They explain what each benchmark measures, prefer visual summaries for quick reading, and keep JSON/CSV outputs under the generated artifact directory instead of the repository root.  For terminology across lattice filters, model reduction, and MIMO systems, see :doc:`../algorithms/concept_map`.

Build the full benchmark tutorial gallery with results:

.. code-block:: bash

   ./scripts/build_docs_with_results.sh

.. toctree::
   :maxdepth: 1

   generated/core_filtering
   generated/model_reduction
   generated/hankel_reduction_speedup
   generated/finite_aak_iir_reduction_speedup
   generated/mimo_hankel_reduction_speedup
   generated/matrix_lattice_runtime
   generated/tangential_schur_mimo
   generated/experimental_mimo_matrix_lattice_realization_sweep
   generated/finite_nehari_rank_sweep
   generated/block_levinson
   generated/adaptive_period_sweep
   generated/echo_metric
