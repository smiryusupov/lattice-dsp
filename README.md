# lattice-dsp

**Efficient stable IIR lattice filters and matrix/MIMO lattice DSP for Python.**

`lattice-dsp` is a focused C++/Python toolkit for lattice-parameterized recursive signal processing.  It is built around reflection/PARCOR coefficients, lattice-ladder realizations, adaptive IIR experiments, AR spectral modeling, and matrix-valued lattice/all-pass examples.

The core idea is simple: direct-form IIR denominator updates can move poles outside the unit circle, while lattice parameterizations give a stability-aware coordinate system.  The package exposes that coordinate system through Python APIs, C++/pybind11 kernels, optional OpenMP batch processing, and tutorial-style Sphinx pages with generated figures and benchmark outputs.

## Why this package exists

Most Python DSP packages are broad.  They provide direct-form filters, SOS filters, FFT-based spectral estimators, room simulation, array processing, or general time-series models.  `lattice-dsp` is intentionally narrower and occupies an uncommon niche: stable IIR lattice coordinates, finite model-reduction diagnostics, and SISO-to-MIMO lattice/state-space examples in one C++/Python workflow.

The rare part is the combination, especially on the MIMO side.  The package places scalar IIR lattice/lattice-ladder primitives next to finite-Hankel SISO reduction, finite-section Nehari/AAK-style diagnostics, MIMO block-Hankel reduction, compiled MIMO state-space simulation, and matrix-lattice all-pass scaffolds.  The public docs describe this as a specialized and rare combination, not as a claim of exclusivity.

The package is centered on:

- stable scalar IIR lattice and lattice-ladder filtering;
- adaptive recursive filtering using bounded reflection-coefficient updates;
- AR, Burg, Levinson-Durbin, and lattice-based spectral diagnostics;
- compiled batch processing for many independent filters/signals;
- a million-sample throughput tutorial showing how compact IIR/lattice models can process long acoustic-like decays with small recursive state;
- a large echo-scale stress tutorial for million-sample signals and high-order stable recursive models, with FIR echo tap-traffic scale estimates;
- matrix/MIMO lattice experiments such as streaming all-pass responses, streaming paraunitary-style analysis, compact unitary responses, multichannel AR models, causal online MIMO lattice prediction, and finite tangential Schur/Pick/J-inner diagnostics.

A distinctive part of the package is the **matrix/MIMO lattice and MIMO model-reduction** direction.  The project includes tutorial examples for matrix all-pass systems with causal online forward processing, streaming paraunitary-style analysis with finite-record adjoint checks, multichannel AR, causal online MIMO lattice prediction, MIMO lattice comparisons, unitary multichannel transforms, coupled block-Hankel reduction, and bridge diagnostics from MIMO Markov data to matrix-lattice all-pass scaffolds.  The docs separate supported finite-dimensional capabilities from diagnostic scaffolds so users can see the intended scope of each API.

### Online versus offline pieces

The runtime filtering APIs are causal, but several estimation and diagnostic APIs are intentionally batch/offline:

| Package area | Online/causal status |
|---|---|
| Scalar `LatticeIIR`, `LatticeLadderIIR`, adaptive NLMS/RLS, and `StreamingBlockProcessor` | Causal stateful IIR processing; samples are consumed left-to-right with no lookahead. |
| `burg_*`, `levinson_durbin_*`, and multichannel/block Levinson AR tools | Batch coefficient estimation from a finite record or autocorrelation sequence; after estimation, the AR/IIR model can be run causally. |
| `MIMOLatticePredictor` | Causal online MIMO lattice prediction from fitted block-Levinson forward/backward reflection matrices; `predict()` uses only previous vectors, then `update(y_t)` consumes the current vector. |
| `mimo_state_space_process_batch` | Causal batched MIMO state-space simulation using the current input and previous state. |
| `MatrixLatticeAllPass.frequency_response`, `OnlineMatrixLatticeAllPass`, and matrix-lattice adjoint diagnostics | Frequency-response diagnostics plus causal sample-by-sample forward all-pass processing. Reconstruction/adjoint demos use finite-record time-domain adjoints and are noncausal/transductive by design. |
| finite-Hankel SISO/MIMO reduction and Nehari/AAK-style diagnostics | Offline preprocessing from impulse/Markov/tail data; the returned reduced state-space or IIR model may then be used causally when applicable. |


### Online MIMO lattice prediction

The package includes a causal online multichannel lattice predictor.  The
coefficient fit may be done offline from a training record, but runtime
prediction is left-to-right: the prediction is requested before the current
vector is observed.

```python
prediction = predictor.predict()   # uses only previous vectors
error = predictor.update(y_t)      # consumes the current vector afterward
```

For diagonal matrix reflections, the MIMO predictor reduces exactly to
independent one-channel/SISO predictors.  With full matrix reflections, it can
model cross-channel dynamics that independent SISO filters cannot use.  The
examples include both sanity checks: diagonal MIMO equals independent SISO, and
coupled online MIMO reduces residual cross-channel correlation on held-out data.


### Tangential Schur and J-inner baseline

The package includes an experimental but conservative tangential-Schur layer for
finite matrix-valued interpolation diagnostics.  It supports right tangential
data

```math
S(z_i)U_i = V_i, \qquad \|S\|_\infty \le 1,
```

the definite Pick matrix

```math
P_{ij}=\frac{U_i^H U_j - V_i^H V_j}{1-\overline{z_i}z_j},
```

constant contractive solutions when the data are compatible with one, and
elementary Potapov--Blaschke/J-inner factors with numerical checks of

```math
\Theta(e^{j\omega})^H J \Theta(e^{j\omega}) = J.
```

The Pick matrix is documented as a finite RKHS/de Branges--Rovnyak Gram matrix,
``P_ij = U_i^H K_S(z_i,z_j) U_j``.  The Potapov factors are documented as
interpolation-side J-inner objects: they are related to stable lossless/all-pass
matrix-lattice systems, while the actual causal signal runtime is provided by
`OnlineMatrixLatticeAllPass`.

This is intentionally a finite definite baseline, not a full generalized
indefinite matrix Schur solver and not the full recursive tangential-Schur
manifold parametrization used by Hanzon--Olivi--Peeters/Marmorat for fixed-degree
lossless systems and balanced realizations.  The goal is to make the
interpolation and J-inner structure behind the MIMO lattice examples inspectable
and testable.  The verification suite checks scalar reductions, random constant
MIMO contractions, diagonal-MIMO direct-sum structure, feasible nonconstant
Blaschke data, infeasible data, unitary congruence of tangential directions,
near-boundary conditioning, and J-inner residuals for Potapov factors/products.

### MIMO verification story

The MIMO examples are organized around explicit checks rather than only visual
demos:

| Claim | Where to look |
|---|---|
| Diagonal MIMO reduces exactly to independent SISO | `mimo_diagonal_equals_independent_siso.py` |
| Online MIMO prediction is causal | `causal_mimo_lattice_prediction.py` |
| Coupled MIMO captures cross-channel history that SISO misses | `online_coupled_mimo_vs_siso.py` |
| Streaming matrix-lattice all-pass matches its frequency response | `matrix_lattice_allpass.py` |
| Paraunitary/ML-style forward transforms are streaming but adjoint checks are finite-record/noncausal | `paraunitary_filter_bank_demo.py`, `ml_unitary_convolution_demo.py` |

The documentation page `docs/algorithms/mimo_verification_map.rst` collects
these claims with their data-use status and diagnostics.

### Causality and data-use terminology

The docs distinguish four cases:

| Term | Meaning in the package |
|---|---|
| Causal runtime | Output or prediction is formed from current/past inputs, previous outputs, or stored state; no future samples are used. |
| Batch/offline estimation | A finite training record, covariance sequence, impulse response, or Markov tensor is used to fit coefficients. |
| Inductive evaluation | Fit on one record, then run causally on a separate held-out stream. |
| Transductive/block diagnostic | A whole finite block is used to choose or apply a transform; useful for diagnostics but not the same as online streaming. |

Prediction uses one signal and asks for ``y_t`` from past ``y`` values.  Echo
cancellation/system-identification examples use two signals: a reference/far-end
input and a microphone/desired signal.  The synthetic echo pages are controlled
metric demonstrations, not production acoustic echo cancellers.

## Current scope

### Available capabilities

- Reflection/PARCOR coefficient conversion for stable IIR denominators.
- Scalar `LatticeIIR` and `LatticeLadderIIR` filtering.
- Adaptive lattice-ladder NLMS and RLS-style examples.
- Adaptive notch tracking and stable recursive system-identification demos.
- H∞/minimax LMS tutorial inspired by Hassibi, Sayed, and Kailath.
- Lattice DSP concept map connecting Schur/Szegő recursions, orthogonal filters, all-pass completions, Hankel forms, Padé approximation, and matrix/MIMO lattice forms.
- Package-positioning, algorithm-selection, and theory-guide pages covering public scope, the package niche, Pick interpolation conditioning, Schur/reflection stability coordinates, inner/outer factorization, Hardy-space intuition, Hankel operators, reachability, observability, and Kronecker finite-rank motivation.
- Dependency-free interoperability recipes for scalar stream shapes, online and batched MIMO conventions, MIMO Markov tensors, MIMO AR coefficients, matrix-lattice reflections, tangential Schur/Pick data, SciPy/state-space/MATLAB bridges, Pyroomacoustics-style room impulse responses, WAV/eSpeak workflows, and optional user-side audio loaders.
- Finite-Hankel singular-value diagnostics and finite-Hankel SISO/MIMO reduction helpers in C++, plus finite Nehari/AAK intuition tutorials, exact rational-tail validation, rational-candidate selection helpers, and amortization benchmarks.
- AR tools: autocorrelation, Levinson-Durbin reflection coefficients, Burg reflection estimates, and spectral diagnostics.
- Multichannel/block Levinson-Durbin utilities, dense block-Toeplitz baselines, and causal online MIMO lattice prediction from matrix reflection coefficients.
- Matrix lattice/all-pass utilities for compact MIMO, causal online forward all-pass filtering, time-domain finite-record adjoint diagnostics, bridge diagnostics, and streaming unitary/paraunitary-style examples.
- Tangential Schur/Pick utilities: right tangential data containers, definite Pick matrices, constant-solution sanity checks, elementary Potapov/J-inner factors, and J-inner residual diagnostics.
- C++/pybind11 extension with optional OpenMP batch processing.
- Sphinx tutorial pages that run examples/benchmarks and embed output, figures, CSV/JSON data, and source code.

### Not the goal

`lattice-dsp` is not a replacement for SciPy, not a production acoustic echo canceller, not a WebRTC/Speex replacement, and not a complete wireless-communications or room-acoustics framework.  It also does not require Pyroomacoustics, eSpeak/eSpeak NG, librosa, or soundfile; those are user-side tools that can provide arrays, WAV files, or room impulse responses.

The repository keeps synthetic ERLE and channel/equalization examples because they are useful ways to test adaptive filters and MIMO signal-processing ideas.  They are controlled DSP demonstrations, not production product claims.

## Install for development

```bash
python -m pip install -e '.[dev,examples,benchmark,docs]'
pytest
```

The extension is built with `scikit-build-core`, CMake, pybind11, NumPy, and optional OpenMP.

## Quick example: stable IIR from reflection coefficients

```python
import numpy as np
from lattice_dsp import LatticeIIR, reflection_to_denominator

rng = np.random.default_rng(0)
x = rng.normal(size=4096)

reflection = [0.35, -0.2, 0.1]
numerator = [0.25, 0.0, 0.15, 0.7]

print(reflection_to_denominator(reflection))

y = LatticeIIR(reflection, numerator).process(x)
```

## Quick example: adaptive stable IIR identification

```python
import numpy as np
from lattice_dsp import AdaptiveLatticeLadderNLMS, LatticeIIR

rng = np.random.default_rng(1)
x = rng.normal(size=8000)

target = LatticeIIR([0.55, -0.32], [0.4, -0.1, 0.65])
desired = np.asarray(target.process(x), dtype=float)

adaptive = AdaptiveLatticeLadderNLMS(
    initial_reflection=[0.0, 0.0],
    initial_taps=[0.0, 0.0, 0.0],
    mu_taps=0.06,
    mu_reflection=0.0015,
    margin=1e-4,
)
error = np.asarray(adaptive.adapt_block(x, desired), dtype=float)

print("learned reflection:", np.round(adaptive.reflection, 4))
print("final MSE:", np.mean(error[-1000:] ** 2))
```

## Finite Nehari / rational-candidate workflow

The package now includes a conservative finite-dimensional workflow for SISO Nehari/AAK-style experiments:

```python
import lattice_dsp as ld

criteria = ld.FiniteNehariCandidateCriteria(
    max_tail_error=1e-3,
    max_rational_error=1e-2,
    max_pole_radius=0.99,
)

candidates = ld.finite_nehari_rational_candidates(
    tail,
    ranks=[1, 2, 3, 4],
    rows=48,
    cols=48,
    criteria=criteria,
)
selected = ld.select_finite_nehari_candidate(candidates)

certificate = ld.finite_aak_siso_certificate(
    tail,
    rank=selected["rank"],
    rows=48,
    cols=48,
    criteria=criteria,
)
print(certificate["sigma_next"])
print(certificate["schmidt_left_residual"], certificate["schmidt_right_residual"])
```

This workflow is documented as finite-section candidate selection and certification, not as a full infinite-dimensional AAK/Nehari solver.  The exact-rational-tail tutorial validates the finite workflow on a known rank-3 exponential tail.  For full SISO lattice/IIR filters, ``finite_aak_reduce_iir`` applies the same workflow to the impulse response and returns a selected reduced numerator, denominator, and reflection coefficients when stable.


## Model-reduction scope and validation

The model-reduction stack is organized by supported scope:

| Layer | Status |
|---|---|
| SISO finite-Hankel reduction | implemented as a C++ finite-section Ho--Kalman baseline |
| Large-order reduction speedup benchmark | implemented, including break-even and end-to-end speedup diagnostics |
| MIMO block-Hankel reduction | implemented as a state-space baseline from Markov matrices, with diagonal/coupled tutorials, a compiled batched state-space processor, and a speed/error benchmark |
| Causal MIMO lattice prediction | implemented as an online vector lattice predictor from block-Levinson forward/backward reflection matrices |
| Matrix-lattice MIMO runtime and bridge diagnostics | implemented as causal streaming forward all-pass filtering, time-domain finite-record adjoint diagnostics, runtime benchmark, and block-Hankel-to-lattice scaffold diagnostic |
| Tangential Schur/Pick/J-inner baseline | implemented for finite definite right tangential data, Pick PSD checks, constant-solution sanity checks, and elementary Potapov factors |
| Finite Nehari/rational candidate workflow | implemented as a finite-dimensional teaching and validation API |
| Finite-section AAK/Nehari certificate | implemented; verifies Schmidt-pair identities and candidate quality |
| Exact rational-tail validation | implemented; known rank-3 tails recover the correct rank and poles |
| Finite-section AAK/Nehari IIR reduction | implemented as a high-level stable SISO IIR reduction candidate |
| Full infinite-dimensional SISO AAK/Nehari solver | out of scope for the current public alpha |
| Matrix AAK/Nehari solver | out of scope for the current public alpha |
| Generalized indefinite matrix Schur synthesis | out of scope for the current public alpha |

Validation coverage for this scope includes the full test suite, release-trust tests for the public claims, rendered tutorial docs, exact rational-tail validation, MIMO diagonal equivalence, and MIMO block-Hankel reduction.

Public validation command:

```bash
python -m pip install -e '.[dev,examples,benchmark,docs]'
./scripts/release_validation.sh
```

For a quicker claim-focused pass during development:

```bash
PYTHONPATH="$PWD" pytest -q tests/test_release_trust_claims.py
```

The release-notes page ``docs/release_notes_0_1.rst`` records the current 0.1
public-alpha validation checklist, public API audit summary, representative SISO
finite-section AAK/Nehari IIR results, and compiled MIMO block-Hankel amortization
numbers. Treat benchmark numbers as local reproducibility examples rather than
universal performance guarantees.

## Tutorial examples

The examples and benchmarks are designed to be read through the Sphinx gallery, not only as raw scripts.  Example pages keep useful captured output; benchmark pages prefer visual summaries generated from JSON/CSV artifacts, with the raw files kept as downloads for reproducibility.

Build the tutorial gallery locally:

```bash
./scripts/build_docs_with_results.sh
xdg-open docs/_build/html/examples/index.html
xdg-open docs/_build/html/benchmarks/index.html
```

For the long-signal speed motivation, run the million-sample throughput tutorial locally:

```bash
python examples/million_sample_iir_throughput.py --samples 1000000 --tail-taps 131072
```

For a larger echo-scale stress case, run:

```bash
python examples/large_order_echo_stress.py --samples 1000000 --order 512 --echo-taps 131072
```

For the MIMO long-signal stress case, run:

```bash
python examples/mimo_long_signal_state_space_stress.py --samples 1000000 --batch 1 --inputs 8 --outputs 8 --full-order 64 --reduced-order 16 --fir-taps 131072
```

New users should start with the recommended learning path and theory guide:

```text
docs/learning_path.rst
docs/theory/choosing_algorithms.rst
docs/theory/hardy_hankel_state_space.rst
docs/interoperability.rst
```

The intended sequence is algorithm selection first, then the interoperability page for the package-wide array-shape conventions and optional bridges to external tools, then scalar stability, adaptive/spectral diagnostics, finite-Hankel reduction, finite-section Nehari/AAK diagnostics, and finally matrix/MIMO baselines, runtime benchmarks, and bridge diagnostics.

Representative tutorials:

### Orientation and theory bridge

- [`algorithm_selection_demo.py`](examples/algorithm_selection_demo.py) — choose the API family by task, diagnostic, and validation scope.
- [`reachability_observability_hankel_demo.py`](examples/reachability_observability_hankel_demo.py) — connect state-space reachability/observability to Hankel singular values.
- [`mimo_long_signal_state_space_stress.py`](examples/mimo_long_signal_state_space_stress.py) — stress finite MIMO block-Hankel reduction plus compiled MIMO state-space runtime on long multichannel signals.

### Interoperability recipes

- [`docs/interoperability.rst`](docs/interoperability.rst) — package-wide shape conventions for scalar streams, online MIMO streams, batched MIMO state-space data, MIMO Markov tensors, MIMO AR coefficients, matrix-lattice reflections, and tangential Schur/Pick data, plus optional SciPy, state-space/control, MATLAB/Octave, Pyroomacoustics, WAV/eSpeak, librosa, and soundfile bridges.
- [`pyroomacoustics_mimo_rir_recipe.py`](examples/pyroomacoustics_mimo_rir_recipe.py) — convert Pyroomacoustics-style `room.rir[mic][source]` data to a MIMO Markov tensor for finite block-Hankel reduction, without importing Pyroomacoustics.
- [`external_audio_wav_recipe.py`](examples/external_audio_wav_recipe.py) — bring eSpeak/eSpeak NG or other external WAV sources into NumPy arrays using standard-library WAV I/O.

### Scalar lattice and adaptive IIR

- [`hinf_lms_reproduction.py`](examples/hinf_lms_reproduction.py) — LMS as a worst-case energy-gain/minimax filter, not only approximate least squares.
- [`reflection_conversion.py`](examples/reflection_conversion.py) — reflection coefficients and denominator conversion.
- [`reflection_coefficients_stability_demo.py`](examples/reflection_coefficients_stability_demo.py) — static stability diagnostics for reflection versus direct denominator parameters.
- [`lattice_ladder_realization.py`](examples/lattice_ladder_realization.py) — lattice-ladder realization of `B(z) / A(z)`.
- [`stability_vs_direct_iir.py`](examples/stability_vs_direct_iir.py) — why bounded lattice updates are safer than unconstrained denominator updates.
- [`adaptive_iir_system_identification.py`](examples/adaptive_iir_system_identification.py) — stable recursive system identification.
- [`tracking_drifting_iir_system.py`](examples/tracking_drifting_iir_system.py) — tracking a slowly changing target IIR system.

### AR and spectral diagnostics

- [`periodogram_vs_ar_spectrum.py`](examples/periodogram_vs_ar_spectrum.py) — periodogram vs Levinson/Burg AR spectra.
- [`capon_spectrum_demo.py`](examples/capon_spectrum_demo.py) — Capon/MVDR spectrum as a high-resolution diagnostic.
- [`spectral_diagnostics_comparison.py`](examples/spectral_diagnostics_comparison.py) — periodogram, AR, Burg, and Capon comparison.
- [`burg_levinson_ar_tools.py`](examples/burg_levinson_ar_tools.py) — AR estimates for a known stable process.
- [`ar_spectral_estimation.py`](examples/ar_spectral_estimation.py) — AR spectral estimation using reflection coefficients.

### Matrix/MIMO lattice direction

- [`matrix_lattice_allpass.py`](examples/matrix_lattice_allpass.py) — matrix reflection coefficients and unitary all-pass response.
- [`coupled_mimo_lattice_filter.py`](examples/coupled_mimo_lattice_filter.py) — causal streaming matrix-lattice all-pass analysis with finite-record adjoint reconstruction diagnostics.
- [`matrix_unitary_response_compression.py`](examples/matrix_unitary_response_compression.py) — compact representation of a frequency-dependent unitary response, plus a causal streaming runtime check.
- [`paraunitary_filter_bank_demo.py`](examples/paraunitary_filter_bank_demo.py) — streaming paraunitary-style analysis with finite-record time-domain adjoint reconstruction.
- [`multichannel_levinson_ar.py`](examples/multichannel_levinson_ar.py) — vector AR prediction and block-Toeplitz comparison.
- [`causal_mimo_lattice_prediction.py`](examples/causal_mimo_lattice_prediction.py) — sample-by-sample causal MIMO lattice prediction from block-Levinson matrix reflections.
- [`matrix_ar_spectral_estimation.py`](examples/matrix_ar_spectral_estimation.py) — spectral-matrix estimation for multichannel AR models.
- [`mimo_lattice_vs_block_levinson.py`](examples/mimo_lattice_vs_block_levinson.py) — side-by-side MIMO lattice and block Levinson diagnostics.
- [`mimo_diagonal_equals_independent_siso.py`](examples/mimo_diagonal_equals_independent_siso.py) — diagonal MIMO as independent SISO filters.
- [`mimo_coupled_model_reduction.py`](examples/mimo_coupled_model_reduction.py) — coupled MIMO block-Hankel reduction with state-space diagnostics.
- [`mimo_model_reduction_stress_cases.py`](examples/mimo_model_reduction_stress_cases.py) — three MIMO stress cases: a 3×3 1/f-type tail, a 10×10 random rational response, and a high-degree ill-conditioned 2×2 response, with H2 error, finite-Hankel spectral-norm error, timing/backend plots, and tangential-Schur/Pick diagnostics.
- [`mimo_hankel_to_matrix_lattice_bridge.py`](examples/mimo_hankel_to_matrix_lattice_bridge.py) — bridge diagnostic from reduced MIMO Markov data to a stable matrix-lattice all-pass scaffold.
- [`experimental_mimo_matrix_lattice_realization.py`](examples/experimental_mimo_matrix_lattice_realization.py) — experimental solver-style all-pass/polar fit from a reduced MIMO state-space model to a matrix-lattice scaffold.
- [`experimental_mimo_matrix_lattice_calibration.py`](examples/experimental_mimo_matrix_lattice_calibration.py) — calibration of static gain compensation around a known matrix-lattice all-pass response.
- [`ml_unitary_convolution_demo.py`](examples/ml_unitary_convolution_demo.py) — ML-inspired streaming norm-preserving multichannel convolution demo with finite-record adjoint check.

## Benchmarks and HTML reports

Benchmark outputs are written under `reports/` or generated Sphinx artifact directories, not the repository root.  The Sphinx benchmark gallery turns benchmark JSON/CSV payloads into PNG summaries so readers can compare speed, error, and quality metrics visually rather than reading console-style JSON tables.

Coupled MIMO block-Hankel reduction benchmark:

```bash
python benchmarks/mimo_hankel_reduction_speedup.py \
  --full-orders 8 16 \
  --reduced-orders 2 4 6 8 \
  --inputs 3 \
  --outputs 3 \
  --batch 8 \
  --samples 6000 \
  --reuse-count 50 \
  --n-threads 1 \
  --output reports/mimo-hankel-reduction-speedup.json
```

The MIMO benchmark separates processing speedup from one-shot end-to-end speedup.
Reduction is a preprocessing step; use `--reuse-count` to report the amortized
end-to-end speedup when the reduced model is reused across many additional batches.


Matrix-lattice all-pass runtime benchmark:

```bash
python benchmarks/matrix_lattice_runtime.py \
  --dims 2 3 4 \
  --orders 2 4 8 \
  --n-freq 1024 \
  --repeats 3 \
  --n-threads 1 \
  --output reports/matrix-lattice-runtime.json
```

This benchmark compares the compiled matrix-lattice frequency-response evaluator
with the NumPy reference path.  It reports runtime speedup, implementation
agreement, and all-pass unitarity error.

MIMO tangential-Schur/Pick and J-inner benchmark:

```bash
python benchmarks/tangential_schur_mimo_benchmark.py \
  --dims 2 3 4 \
  --points 4 8 \
  --diagonal-points-per-channel 4 8 \
  --boundary-grid 256 \
  --repeats 2 \
  --output reports/tangential-schur-mimo.json \
  --csv-output reports/tangential-schur-mimo.csv
```

This benchmark connects the tangential-Schur layer to the MIMO story.  It
measures finite right-tangential Pick construction, PSD/eigenvalue checks,
constant Schur recovery, Potapov/J-inner boundary diagnostics, and the diagonal
MIMO reduction to independent scalar Pick blocks.

Experimental MIMO matrix-lattice realization diagnostic sweep:

```bash
python benchmarks/experimental_mimo_matrix_lattice_realization_sweep.py \
  --full-order 12 \
  --reduced-orders 4 6 8 \
  --lattice-orders 2 4 6 \
  --channels 3 \
  --n-freq 192 \
  --n-markov 192 \
  --output reports/experimental-mimo-matrix-lattice-realization-sweep.json
```

This sweep reports polar-factor fit error, raw state-response error, static-gain
compensated error, gain-condition diagnostics, and a qualitative classification.
It is intentionally an experimental all-pass/polar scaffold benchmark, not a
matrix AAK/Nehari optimality result.

Core filtering benchmark:

```bash
python benchmarks/run_benchmarks.py \
  --channels 64 \
  --samples 50000 \
  --repeats 5 \
  --output reports/benchmark-results.json
```

Model-reduction benchmark:

The baseline benchmark compares simple stable reductions.  The C++ backend also exposes finite-Hankel singular-value diagnostics and a finite-Hankel/Ho--Kalman reducer.  The docs are explicit that this is a finite-section approximation, not an exact infinite-dimensional AAK/Nehari solver.

```bash
python benchmarks/model_reduction_benchmark.py \
  --full-order 16 \
  --orders 2 4 6 8 12 16 \
  --channels 64 \
  --samples 50000 \
  --repeats 5 \
  --output reports/model-reduction-results.json
```

Finite-Hankel reduction amortization benchmark:

```bash
python benchmarks/hankel_reduction_speedup.py \
  --full-orders 16 32 64 \
  --reduced-orders 4 8 12 16 \
  --channels 64 \
  --samples 50000 \
  --repeats 3 \
  --output reports/hankel-reduction-speedup.json
```

This benchmark reports reduction time, full/reduced filtering time, filter speedup, SNR/error, and the break-even sample count where the one-time reduction cost pays for itself.

Finite-section AAK/Nehari IIR reduction comparison benchmark:

```bash
python benchmarks/finite_aak_iir_reduction_speedup.py \
  --full-orders 8 16 32 \
  --target-orders 3 4 6 8 12 \
  --channels 32 \
  --samples 30000 \
  --repeats 3 \
  --output reports/finite-aak-iir-reduction-speedup.json
```

This benchmark compares the finite-Hankel/Ho--Kalman baseline with the finite-section AAK/Nehari candidate workflow on the same stable SISO IIR filters.  It reports reduction time, filtering speedup, end-to-end speedup including reduction, output SNR, magnitude-response error, stability, and break-even samples per channel.

A representative local run on 32 channels and 30,000 samples/channel showed the current practical trade-off:

| Method | Full order | Reduced order | Filter speedup | End-to-end speedup | Output SNR | Max magnitude error | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| finite-Hankel | 32 | 3 | 11.6x | 4.2x | 51.4 dB | 0.210 dB | fastest useful baseline |
| finite-Hankel | 32 | 6 | 2.8x | 2.1x | 143.7 dB | ~0 dB | high-quality baseline |
| finite-AAK candidate | 32 | 6 | 3.8x | 1.3x | 113.0 dB | ~0 dB | certificate/selection path with useful speedup |
| finite-AAK candidate | 32 | 8 | 2.4x | 1.2x | 157.1 dB | ~0 dB | higher-quality candidate |

The benchmark takeaway is that finite-Hankel is the supported fast baseline, while the finite-section AAK/Nehari candidate workflow is valuable for rank selection, certificates, rational diagnostics, and selected high-quality reductions.  The benchmark documents that trade-off directly.

Adaptive update-period sweep:

```bash
python benchmarks/adaptive_period_sweep.py \
  --periods 1 2 4 8 16 32 \
  --samples 20000 \
  --repeats 5 \
  --output reports/adaptive-period-sweep.json \
  --csv-output reports/adaptive-period-sweep.csv
```

Build rendered benchmark/tutorial pages, including benchmark plots:

```bash
./scripts/build_docs_with_results.sh
```

Generated figures, JSON, CSV, and HTML stay under `docs/examples/generated/`, `docs/benchmarks/generated/`, `docs/_build/`, or `reports/`. Source distributions contain package code, tests, hand-written docs, and reproducible generation scripts rather than local cache/build outputs.

## Core API sketch

```python
from lattice_dsp import (
    AdaptiveLatticeLadderNLMS,
    AdaptiveNotch,
    LatticeIIR,
    LatticeLadderIIR,
    LatticeLadderNLMS,
    LatticeLadderRLS,
    MatrixLatticeAllPass,
    autocorrelation,
    block_levinson_durbin,
    bounded_reflection_from_raw,
    burg_reflection,
    contractive_matrix_from_raw,
    denominator_to_reflection,
    ladder_to_numerator,
    multichannel_autocorrelation,
    numerator_to_ladder,
    process_batch,
    reflection_to_denominator,
    rls_process_batch,
    solve_block_yule_walker_direct,
)
```

## Documentation

Build the full Sphinx docs with generated results:

```bash
python -m pip install -e '.[dev,examples,benchmark,docs]'
./scripts/build_docs_with_results.sh
xdg-open docs/_build/html/index.html
```

The docs cover:

- an algorithm-selection guide for choosing APIs by task, input data, and validation scope;
- a Hardy/Hankel/state-space guide covering causal Hardy-space intuition, Hankel operators, reachability, and observability;
- a concept map linking Schur recursion, Szegő polynomials, Christoffel-Darboux identities, orthogonal filters, all-pass completions, Hankel/Padé approximation, and matrix/MIMO lattice forms;
- scalar lattice and lattice-ladder IIR algorithms;
- NLMS/RLS adaptive filtering and the H∞/minimax interpretation of LMS;
- scalar Burg/Levinson-Durbin and multichannel/block Levinson AR tools;
- periodogram, AR, Burg, and Capon spectral diagnostics;
- matrix/MIMO lattice all-pass systems;
- causal coupled matrix-lattice filtering and block-Hankel-to-lattice bridge diagnostics;
- streaming paraunitary-style analysis, unitary-convolution, and multichannel decorrelation examples;
- benchmarks and interpretation guidelines;
- API reference and further reading.

## Concept map for the theory

A useful way to read the package is through the theory guide and concept map: `docs/theory/choosing_algorithms.rst`, `docs/theory/filtering_relationships.rst`, `docs/theory/hardy_hankel_state_space.rst`, and `docs/algorithms/concept_map.rst`.  Together, they explain which algorithm to choose and how the main mathematical objects connect:

```text
filtering relationships: equalization / echo cancellation / prediction
→ stable recursive filters
→ Schur / Szegő recursions
→ reflection coefficients and prediction-error filters
→ orthogonal and all-pass completions
→ Hankel / Padé / Nehari / AAK model-reduction theory
→ matrix/MIMO lattice systems where scalar coefficients become contractions
```

This page is intended for users who recognize terms like Schur recursion, Szegő polynomials, Christoffel-Darboux formula, Padé approximation, or Schmidt pairs, but want to know exactly where those ideas appear in the examples.

## Model-reduction direction

`lattice-dsp` now includes a first finite-Hankel SISO reduction path in the C++ backend:

```python
from lattice_dsp import finite_hankel_reduce_iir

result = finite_hankel_reduce_iir(
    reflection=[0.62, -0.48, 0.36, -0.28, 0.20, -0.14, 0.09, -0.05],
    numerator=[1.0, -0.22, 0.15, 0.08, -0.05, 0.03, 0.0, 0.0, 0.0],
    reduced_order=4,
    n_impulse=360,
    rows=48,
    cols=48,
)

print(result["denominator"])
print(result["retained_hankel_energy"], result["relative_impulse_error"], result["stable"])
```

This is documented as **finite-Hankel/Ho--Kalman** reduction.  It builds truncated Hankel matrices from an impulse response, uses the leading singular directions to form a lower-order realization, and returns numerator/denominator/reflection diagnostics.  Exact infinite-dimensional Nehari/AAK optimality is outside the current public scope.  The old `finite_hankel_aak_reduce_*` names remain as compatibility aliases, but the preferred public names are `finite_hankel_reduce_*`.  The package also exposes `finite_nehari_approximate_tail`, a finite-dimensional teaching helper for anticausal tails; the Nehari/AAK tutorial, rank-sweep benchmark, rational-tail bridge, Schmidt-pair diagnostic page, and candidate-selection API use it to show the finite Hankel singular-value picture behind the theory while presenting it as a finite-section workflow rather than a full infinite-dimensional solver.

The same finite-Hankel baseline is also available for MIMO Markov parameters:

```python
from lattice_dsp import (
    finite_hankel_reduce_mimo,
    mimo_state_space_markov_response,
    mimo_state_space_process_batch,
)

# markov has shape (samples, outputs, inputs)
result = finite_hankel_reduce_mimo(
    markov,
    reduced_order=6,
    block_rows=24,
    block_cols=24,
)

reduced_markov = mimo_state_space_markov_response(
    result["A"], result["B"], result["C"], result["D"], markov.shape[0]
)

# x has shape (batch, samples, inputs); y has shape (batch, samples, outputs).
y = mimo_state_space_process_batch(result["A"], result["B"], result["C"], result["D"], x)
```

For MIMO systems the reduced model is returned as state-space matrices `A, B, C, D`, not as scalar numerator/denominator coefficients.  This provides a reference block-Hankel baseline that is separate from matrix Nehari/AAK methods.  `mimo_state_space_process_batch` is a compiled C++/OpenMP runtime helper for reusing either the full or reduced state-space model on batched multichannel signals.  The `mimo_model_reduction_stress_cases.py` tutorial exercises this baseline on a slow 1/f-type matrix tail, a 10×10 random rational response, and a high-degree ill-conditioned 2×2 response; the tutorial also reports finite-Hankel spectral-norm tail error and timing/backend labels; the tangential-Schur/Pick layer is used there as a finite sampled interpolation certificate rather than as a full reduction solver.

The matrix-lattice bridge tutorial uses reduced Markov data to initialize a stable all-pass scaffold and compares it with the reduced model's frequency-response polar factor.  The experimental realization helper promotes that diagnostic into a small solver-style API that searches over Markov-initialized matrix-lattice candidates and returns the best all-pass/polar fit.  This is deliberately not presented as an exact realization of arbitrary MIMO gain responses: `MatrixLatticeAllPass` is unitary/all-pass.

## Scope boundaries

Included intentionally:

- Stable IIR lattice/lattice-ladder DSP as the core public API.
- Finite-Hankel SISO reduction and finite-section Nehari/AAK-style diagnostics as finite-dimensional model-reduction workflows.
- MIMO block-Hankel reduction, compiled MIMO state-space simulation, stress-case tutorials, and matrix-lattice bridge diagnostics as a specialized MIMO baseline layer.
- Matrix/MIMO lattice examples that are tutorial-driven and tied to explicit diagnostics.
- Reproducible benchmark and example outputs through Sphinx-generated pages.
- Synthetic AEC/channel examples framed as metrics and controlled identification demos only.

Outside the current public scope:

- Matrix-lattice fitting/identification for measured multichannel responses beyond the current scaffold diagnostic.
- Full matrix-valued AAK/Nehari optimal reduction.
- Exact dynamic matrix-lattice realization of arbitrary nonunitary MIMO gain responses.
- Complete wireless, room-acoustics, or acoustic echo-cancellation systems.
- Fast transversal, QR-lattice, and long-filter production variants.
- Production guidance that decides when compact IIR/lattice models should replace long FIR models in a deployed product.


The model-reduction tutorials also include `examples/aak_siso_candidate_selection.py`, which selects a finite SISO AAK/Nehari rational candidate by rank using tail error, rational realization error, and pole-radius thresholds.  The reusable workflow is available as `finite_nehari_rational_candidates(...)` plus `select_finite_nehari_candidate(...)`.  `examples/aak_siso_certificate_demo.py` then uses `finite_aak_siso_certificate(...)` to verify the finite Schmidt-pair identities and attach the rational candidate for the same rank.  The high-level `finite_aak_reduce_tail(...)` and `finite_aak_reduce_iir(...)` helpers wrap those pieces for finite-section tail and stable-IIR reduction experiments.
