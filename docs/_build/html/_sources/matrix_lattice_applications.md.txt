# Matrix lattice application examples

The matrix-lattice layer is deliberately general.  It is not a 5G or wireless
framework.  It provides compact, stable, matrix-valued all-pass primitives whose
frequency response is unitary when every matrix reflection coefficient is a
strict contraction.

This makes it useful anywhere a multichannel transform should be stable,
invertible, and energy preserving.

## Examples

```bash
python examples/matrix_unitary_response_compression.py
python examples/paraunitary_filter_bank_demo.py
python examples/ml_unitary_convolution_demo.py
python examples/multichannel_audio_decorrelator.py
```

## Compact unitary response representation

`matrix_unitary_response_compression.py` compares storing a dense complex matrix
at every frequency bin with storing a low-order matrix-lattice representation.
This is a generic MIMO/unitary-response compression demo.  The same idea can
appear in array processing, filter banks, communications, and learned
frequency-dependent transforms.

## Paraunitary / perfect reconstruction transforms

`paraunitary_filter_bank_demo.py` uses the causal online matrix-lattice runtime
for the forward analysis transform.  The corresponding reconstruction check uses
a finite-record adjoint of the measured/truncated impulse response.  The forward
path is streaming and causal; the adjoint reconstruction is a finite-record
diagnostic and is generally noncausal because it depends on the whole transformed
record.

## ML-adjacent unitary convolution blocks

`ml_unitary_convolution_demo.py` demonstrates a norm-preserving multichannel
convolution-like transform using causal forward processing plus a finite-record
adjoint diagnostic.  Orthogonal/unitary layers are useful in ML because they can
keep forward and adjoint maps well-conditioned.  This example is a NumPy
primitive demo, not a training framework, and the adjoint/reconstruction step is
not presented as an online inverse filter.

## Multichannel audio decorrelation

`multichannel_audio_decorrelator.py` uses a real-coefficient matrix-lattice
all-pass filter to reduce zero-lag inter-channel correlation while preserving
total energy.  This is useful for spatial-audio and multichannel decorrelation
experiments.

## Scope

Good package scope:

- matrix reflection coefficients and stability checks;
- causal streaming matrix-lattice forward processing;
- unitary/paraunitary frequency-response evaluation;
- finite-record adjoint diagnostics for reconstruction tests;
- compact matrix-valued all-pass representations;
- examples for multichannel DSP and ML-stable transforms.

Out of scope for the core package:

- complete 5G/MIMO-OFDM simulators;
- production acoustic echo cancellation;
- end-to-end neural-network training systems.
