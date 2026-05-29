# API reference overview

This is a lightweight hand-written API map until generated documentation is set up.

## Core filters

- `LatticeIIR(reflection, numerator)`: stable reflection-parameterized IIR with direct-form reference processing.
- `LatticeLadderIIR(reflection, ladder_taps)`: true synthesis lattice-ladder realization.
- `process_batch(reflection, numerator, x, realization="lattice")`: independent row-wise batch processing, OpenMP-enabled when available.

## Adaptive filters

- `LatticeLadderNLMS`: fixed-denominator adaptive numerator/ladder NLMS.
- `AdaptiveLatticeLadderNLMS`: adaptive numerator and bounded reflection update.
- `adaptive_process_batch(...)`: independent adaptive trials, OpenMP-enabled across rows.
- `AdaptiveNotch`: adaptive second-order notch prototype.

## Conversion and diagnostics

- `reflection_to_denominator`
- `denominator_to_reflection`
- `numerator_to_ladder`
- `ladder_to_numerator`
- `bounded_reflection_from_raw`
- `denominator_reflection_jacobian`
- `denominator_raw_jacobian`
- `denominator_raw_jacobian_finite_difference`

## Tuning

- `tune_reflection_update_period`: choose a speed/quality point for adaptive reflection updates from validation signals.

## Synthetic signals and echo metrics

- `generate_echo_problem(...)`: reproducible synthetic reference/microphone/clean-target arrays for controlled echo-like experiments.
- `EchoProblem`: returned dataclass.
- `iir_filter(...)`: NumPy reference IIR implementation used by the synthetic generator.
- `echo_metrics(microphone, enhanced, clean_target)`: returns synthetic ERLE, MSE improvement, residual power, and segmental ERLE values.
- `erle_db(before_error, after_error)`: computes echo/error power reduction.
- `segmental_erle_db(...)`: returns frame-wise ERLE values.

## Small application helpers

- `HybridEchoCanceller`: adaptive linear lattice/IIR stage plus optional residual processor.  This is a synthetic/demo helper, not production AEC.
- `EchoCancellationResult`: output container with echo estimate, residual, enhanced residual, final filter parameters, and metrics.
- `ResidualAttenuator`: fixed-gain residual attenuator used as a minimal baseline.
- `SpectralResidualSuppressor`: deterministic STFT-domain suppressor for dependency-free experiments.
- `make_residual_processor_from_model`: wrap user objects with `predict` or callable protocols.
- `mse`, `improvement_db`: small metric helpers.

## AR estimation utilities

- `autocorrelation(x, max_lag, biased=True)`
- `levinson_durbin_reflection(autocorr, order, regularization=1e-12)`
- `levinson_durbin_denominator(autocorr, order, regularization=1e-12)`
- `levinson_durbin_error(autocorr, order, regularization=1e-12)`
- `burg_reflection(x, order, regularization=1e-12)`
- `burg_denominator(x, order, regularization=1e-12)`

## RLS adaptive filtering

- `LatticeLadderRLS(reflection, initial_taps, forgetting_factor=0.995, initial_inverse_covariance=1000.0)`
- `rls_process_batch(reflection, initial_taps, x, desired, ...)`

## Streaming helpers

- `BlockProcessor`
- `AdaptiveBlockProcessor`
- `BlockResult`


## Matrix lattice / MIMO all-pass

- `MatrixLatticeAllPass(reflections, residue=None, margin=1e-9, project=False)`
- `contractive_matrix_from_raw(raw, margin=1e-6)`
- `project_matrix_reflection(matrix, margin=1e-6)`
- `is_matrix_reflection_stable(matrix, margin=1e-9)`
- `matrix_lattice_stage_blocks(reflection, margin=1e-9, project=False)`
- `unitary_polar_factor(matrix)`
- `psd_matrix_sqrt(matrix)`
- `_core.matrix_lattice_frequency_response(stage_blocks, residue, omega, n_threads=0)`

The C++ frequency-response function is normally reached through
`MatrixLatticeAllPass.frequency_response()`.
