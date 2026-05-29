Examples tutorials
==================

These pages are generated from the runnable scripts in ``examples/``.  Each tutorial page explains the context, shows the main equations, states what the example verifies when applicable, embeds generated figures/data, and includes the source code.  For terminology across the package, start with :doc:`../algorithms/concept_map`; for MIMO claims, see :doc:`../algorithms/mimo_verification_map`.

Build the full tutorial gallery with results:

.. code-block:: bash

   ./scripts/build_docs_with_results.sh

Orientation tutorials
---------------------

.. toctree::
   :maxdepth: 1

   generated/algorithm_selection_demo

Interoperability recipes
------------------------

.. toctree::
   :maxdepth: 1

   generated/pyroomacoustics_mimo_rir_recipe
   generated/external_audio_wav_recipe

Core scalar lattice tutorials
-----------------------------

.. toctree::
   :maxdepth: 1

   generated/reflection_conversion
   generated/lattice_ladder_realization
   generated/reflection_coefficients_stability_demo
   generated/stability_vs_direct_iir
   generated/openmp_batch_processing
   generated/million_sample_iir_throughput
   generated/large_order_echo_stress

Model-reduction tutorials
-------------------------

.. toctree::
   :maxdepth: 1

   generated/mimo_long_signal_state_space_stress
   generated/reachability_observability_hankel_demo
   generated/finite_hankel_model_reduction
   generated/nehari_aak_siso_toy
   generated/finite_nehari_rational_bridge
   generated/finite_nehari_exact_rational_tail
   generated/aak_siso_schmidt_pair_demo
   generated/aak_siso_certificate_demo
   generated/aak_siso_candidate_selection
   generated/finite_aak_noisy_tail_demo
   generated/finite_aak_iir_reduction_demo
   generated/mimo_finite_hankel_model_reduction
   generated/mimo_coupled_model_reduction
   generated/mimo_model_reduction_stress_cases
   generated/mimo_hankel_to_matrix_lattice_bridge
   generated/experimental_mimo_matrix_lattice_realization
   generated/experimental_mimo_matrix_lattice_calibration

Adaptive and AR tutorials
-------------------------

.. toctree::
   :maxdepth: 1

   generated/adaptive_iir_system_identification
   generated/tracking_drifting_iir_system
   generated/adaptive_batch_processing
   generated/tune_reflection_update_period

Applications and signal-model tutorials
---------------------------------------

.. toctree::
   :maxdepth: 1

   generated/adaptive_notch_tracking
   generated/adaptive_prediction_ar
   generated/burg_levinson_ar_tools
   generated/ar_spectral_estimation
   generated/rls_lattice_identification
   generated/streaming_block_processing
   generated/channel_equalization_toy
   generated/system_identification

Spectral diagnostic tutorials
-----------------------------

.. toctree::
   :maxdepth: 1

   generated/periodogram_vs_ar_spectrum
   generated/capon_spectrum_demo
   generated/spectral_diagnostics_comparison

Adaptive and robust filtering tutorials
---------------------------------------

.. toctree::
   :maxdepth: 1

   generated/hinf_lms_reproduction

Multichannel and matrix tutorials
---------------------------------

.. toctree::
   :maxdepth: 1

   generated/mimo_diagonal_equals_independent_siso
   generated/causal_mimo_lattice_prediction
   generated/online_coupled_mimo_vs_siso
   generated/multichannel_levinson_ar
   generated/matrix_ar_spectral_estimation
   generated/mimo_lattice_vs_block_levinson
   generated/matrix_lattice_allpass
   generated/coupled_mimo_lattice_filter
   generated/multichannel_audio_decorrelator
   generated/matrix_unitary_response_compression
   generated/paraunitary_filter_bank_demo
   generated/ml_unitary_convolution_demo

Tangential Schur and J-inner tutorials
--------------------------------------

.. toctree::
   :maxdepth: 1

   generated/tangential_schur_pick_jinner
   generated/diagonal_tangential_schur_equals_scalar

Synthetic metric tutorials
--------------------------

.. toctree::
   :maxdepth: 1

   generated/echo_cancellation_erle_demo
