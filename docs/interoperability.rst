Interoperability recipes
=======================

``lattice-dsp`` is intentionally dependency-light.  The core package does not
import SciPy, python-control, MATLAB engines, Pyroomacoustics, eSpeak/eSpeak NG,
librosa, soundfile, or audio-codec libraries.  Interoperability is by data
convention: convert external data to NumPy arrays, keep the dimensions explicit,
and then pass those arrays to the lattice, adaptive, AR, model-reduction,
matrix-lattice, or tangential-Schur APIs.

This page collects the conventions used across the package.  Treat it as a
recipe sheet, not as a required integration layer.

Array conventions at a glance
-----------------------------

.. list-table:: Common shapes
   :header-rows: 1
   :widths: 28 32 40

   * - Object
     - Shape
     - Meaning
   * - Scalar stream
     - ``(samples,)``
     - One real-valued SISO signal.
   * - Batched scalar streams
     - ``(batch, samples)`` or ``(signals, samples)``
     - Independent scalar filters/signals processed row-by-row.
   * - Online MIMO stream
     - ``(samples, channels)``
     - One vector sample per row for causal multichannel predictors and matrix-lattice filters.
   * - Batched MIMO state-space input
     - ``(batch, samples, inputs)``
     - Independent multichannel records processed by one state-space model.
   * - Batched MIMO state-space output
     - ``(batch, samples, outputs)``
     - Output returned by the compiled batched state-space runtime.
   * - MIMO Markov / impulse response
     - ``(taps, outputs, inputs)``
     - ``H[k]`` maps input vector ``u[n-k]`` to output vector ``y[n]``.
   * - Multichannel autocorrelation
     - ``(order + 1, channels, channels)``
     - ``R[k] = E{x[n] x[n-k]^H}`` for vector AR tools.
   * - MIMO AR coefficients
     - ``(order, channels, channels)``
     - ``x[n] + sum_j A[j] @ x[n-j-1] = e[n]``.
   * - Matrix-lattice reflections
     - iterable of ``(dim, dim)`` matrices
     - Strict-contraction matrix reflection coefficients.
   * - Matrix-lattice stage blocks
     - ``(stages, 4, dim, dim)``
     - Four block matrices per lattice/J-inner section.
   * - Right tangential Schur points
     - ``(points,)``
     - Interpolation nodes inside the unit disk.
   * - Right tangential directions/values
     - ``(points, dim)`` or ``(points, dim, multiplicity)``
     - Rank-one or higher-multiplicity data for ``S(z_i) U_i = V_i``.

Scalar stream conventions
-------------------------

Scalar filtering APIs consume one-dimensional arrays:

.. code-block:: python

   import numpy as np
   import lattice_dsp as ld

   x = np.asarray(x, dtype=float)          # shape: (samples,)
   reflection = np.array([0.35, -0.2])     # shape: (order,)
   numerator = np.array([0.45, 0.05, 0.2]) # shape: (order + 1,)

   filt = ld.LatticeIIR(reflection, numerator)
   y = np.asarray(filt.process(x), dtype=float)  # shape: (samples,)

Rows are used for independent scalar batches:

.. code-block:: python

   x_batch = np.asarray(x_batch, dtype=float)  # shape: (batch, samples)
   y_batch = ld.process_batch(reflection, numerator, x_batch)
   # y_batch.shape == x_batch.shape

The adaptive batch helpers follow the same row-major convention for independent
scalar problems.  For example, ``adaptive_process_batch`` and
``rls_process_batch`` expect ``x`` and ``desired`` with shape
``(batch, samples)`` and return row-aligned output/error arrays plus final
parameters.

Online MIMO stream conventions
------------------------------

Online multichannel APIs use one row per vector sample:

.. code-block:: python

   x = np.asarray(x, dtype=float)  # shape: (samples, channels)

For matrix-lattice all-pass filtering:

.. code-block:: python

   reflections = [0.2 * np.eye(x.shape[1]), -0.1 * np.eye(x.shape[1])]
   lattice = ld.MatrixLatticeAllPass(reflections)
   online = lattice.to_online_filter()
   y = online.process(x, drain=16)  # shape: (samples + 16, channels)

The ``drain`` argument appends zero-vector inputs after the record so that the
causal IIR tail can be inspected.  The forward processing is causal: output row
``n`` uses current and previous input/state only.

For multichannel lattice prediction:

.. code-block:: python

   r = ld.multichannel_autocorrelation(x, order=4)
   ar = ld.block_levinson_durbin(r, order=4)
   predictor = ld.MIMOLatticePredictor.from_levinson(ar)
   prediction, error = predictor.process(x)
   # prediction.shape == error.shape == x.shape

``prediction[n]`` is formed before row ``x[n]`` is consumed; ``error[n]`` is the
forward prediction error after the update.

Batched MIMO state-space conventions
------------------------------------

The compiled state-space runtime uses a control-style discrete-time realization

``x_state[n+1] = A x_state[n] + B u[n]``

``y[n] = C x_state[n] + D u[n]``

with these shapes:

.. code-block:: python

   A.shape == (state_order, state_order)
   B.shape == (state_order, inputs)
   C.shape == (outputs, state_order)
   D.shape == (outputs, inputs)
   u.shape == (batch, samples, inputs)

   y = ld.mimo_state_space_process_batch(A, B, C, D, u)
   # y.shape == (batch, samples, outputs)

Use ``batch=1`` when you have a single multichannel record but still want the
compiled batched state-space path:

.. code-block:: python

   u_one = u_stream[None, :, :]  # (1, samples, inputs)
   y_one = ld.mimo_state_space_process_batch(A, B, C, D, u_one)[0]

MIMO Markov tensor conventions
------------------------------

MIMO model-reduction helpers use Markov parameters / impulse responses with
shape ``(taps, outputs, inputs)``.  Entry ``markov[k, y, u]`` is the output in
channel ``y`` at delay ``k`` caused by a unit impulse in input channel ``u``.

.. code-block:: python

   markov = ld.mimo_state_space_markov_response(A, B, C, D, n_samples=256)
   # markov.shape == (256, outputs, inputs)

   result = ld.finite_hankel_reduce_mimo(
       markov,
       reduced_order=12,
       block_rows=32,
       block_cols=32,
   )

   reduced_markov = ld.mimo_state_space_markov_response(
       result["A"], result["B"], result["C"], result["D"], markov.shape[0]
   )

This convention is also the bridge format for room impulse responses, state-space
systems, matrix-lattice impulse responses, and finite-record adjoint diagnostics.

MIMO AR coefficient conventions
-------------------------------

The vector AR tools use the convention

``x[n] + A[0] @ x[n-1] + ... + A[p-1] @ x[n-p] = e[n]``.

The sample matrix is row-major in time:

.. code-block:: python

   x.shape == (samples, channels)
   r = ld.multichannel_autocorrelation(x, order=p)
   # r.shape == (p + 1, channels, channels)

   direct = ld.solve_block_yule_walker_direct(r, order=p)
   levinson = ld.block_levinson_durbin(r, order=p)

   direct.coefficients.shape == (p, channels, channels)
   levinson.reflection.shape == (p, channels, channels)
   levinson.backward_reflection.shape == (p, channels, channels)

For residual checks, ``multichannel_prediction_error`` returns
``(samples - order, channels)`` because the first ``order`` rows are needed as
history.

Matrix-lattice reflection shapes
--------------------------------

Matrix-lattice helpers use one square complex reflection matrix per section.
Each reflection should be a strict spectral-norm contraction, the matrix analogue
of scalar ``|k| < 1``:

.. code-block:: python

   reflections = [
       0.20 * np.eye(dim, dtype=np.complex128),
       np.array([[0.05, 0.02], [-0.01, 0.04]], dtype=np.complex128),
   ]

   for k in reflections:
       assert ld.is_matrix_reflection_stable(k)

   lattice = ld.MatrixLatticeAllPass(reflections)
   stage_blocks = lattice.stage_blocks
   residue = lattice.residue

   stage_blocks.shape == (len(reflections), 4, dim, dim)
   residue.shape == (dim, dim)

Frequency-response evaluation uses one-dimensional frequencies in radians/sample:

.. code-block:: python

   omega = np.linspace(0.0, np.pi, 512)
   response = ld.matrix_lattice_frequency_response(stage_blocks, residue, omega)
   # response.shape == (omega.size, dim, dim)

The time-domain impulse response convention is again ``(taps, outputs, inputs)``.
The finite-record adjoint helper consumes ``y.shape == (samples, outputs)`` and
an impulse response ``h.shape == (taps, outputs, inputs)`` and returns an adjoint
record with shape ``(output_length, inputs)``.  That adjoint is a finite-record
diagnostic and is generally noncausal as a reconstruction operation.

Right tangential Schur/Pick data shapes
---------------------------------------

Right tangential data represent conditions

``S(z_i) U_i = V_i``

for a matrix-valued Schur function ``S``.  Points are one-dimensional complex
numbers strictly inside the unit disk.  Directions live in the input space and
values live in the output space.

Rank-one shorthand:

.. code-block:: python

   points = np.array([0.0, 0.25 + 0.1j])      # shape: (points,)
   directions = np.array([[1.0, 0.0], [0.0, 1.0]])
   values = 0.5 * directions
   # directions.shape == values.shape == (points, dim)

   data = ld.RightTangentialSchurData(points, directions, values)
   pick = ld.right_tangential_pick_matrix(data)

Higher-multiplicity data use a third axis:

.. code-block:: python

   directions.shape == (points, input_dim, multiplicity)
   values.shape == (points, output_dim, multiplicity)

The Pick matrix has shape ``(total_conditions, total_conditions)``, where
``total_conditions`` is the sum of the multiplicities over all points.

SciPy filter-design/reference bridge
------------------------------------

SciPy is not required by ``lattice-dsp``, but it is useful as a reference for
standard transfer-function operations.  The denominator convention is compatible
with SciPy's ``a`` vector: a monic denominator ``[1, a1, ..., aN]``.

.. code-block:: python

   import numpy as np
   from scipy import signal
   import lattice_dsp as ld

   b, a = signal.butter(4, 0.2)          # optional SciPy design step
   reflection = ld.denominator_to_reflection(a)

   y_scipy = signal.lfilter(b, a, x)
   y_lattice = ld.LatticeIIR(reflection, b).process(x)
   np.testing.assert_allclose(y_lattice, y_scipy, atol=1e-10)

The reverse bridge is also useful when you want a SciPy reference response from
stable reflection coordinates:

.. code-block:: python

   a = ld.reflection_to_denominator(reflection)
   w, h = signal.freqz(b, a, worN=512)

State-space/control-style matrix bridge
---------------------------------------

Control and signal-processing libraries often expose systems as ``A, B, C, D``.
The package convention is discrete-time and uses ``z^-1``:

``H(z) = D + C z^-1 (I - A z^-1)^-1 B``.

Use the NumPy helper for reference frequency responses and the compiled helper
for long batched streams:

.. code-block:: python

   omega = np.linspace(0.0, np.pi, 512)
   H = ld.mimo_state_space_frequency_response(A, B, C, D, omega)
   markov = ld.mimo_state_space_markov_response(A, B, C, D, n_samples=256)
   y = ld.mimo_state_space_process_batch(A, B, C, D, u_batch)

When importing from another control library, check whether its discrete-time
frequency-response convention uses ``z`` or ``z^-1`` and whether its matrix
orientation is output-by-input.  The ``D`` matrix should have shape
``(outputs, inputs)`` in ``lattice-dsp``.

MATLAB/Octave ``.mat`` exchange
-------------------------------

Use SciPy's ``scipy.io`` in application code when you need MATLAB or Octave
exchange.  Keep shape names explicit so the orientation survives the round trip:

.. code-block:: python

   from scipy.io import savemat, loadmat

   savemat("mimo_reduction.mat", {
       "A": result["A"],
       "B": result["B"],
       "C": result["C"],
       "D": result["D"],
       "markov": markov,  # (taps, outputs, inputs)
   })

   data = loadmat("mimo_reduction.mat", squeeze_me=False)
   markov = np.asarray(data["markov"], dtype=float)

MATLAB is column-major internally, but the stored array dimensions are preserved.
The main practical risk is accidental transposition when switching between
``(samples, channels)`` and ``(channels, samples)`` conventions.

Pyroomacoustics room impulse responses
--------------------------------------

Pyroomacoustics can generate room impulse responses between sources and
microphones.  ``lattice-dsp`` can then treat those impulse responses as a finite
sequence of MIMO Markov matrices.

The usual mapping is:

.. code-block:: text

   room.rir[microphone_index][source_index][tap_index]
        -> markov[tap_index, microphone_index, source_index]

A dependency-free adapter can live in your application or notebook:

.. code-block:: python

   import numpy as np

   def markov_from_pyroom_rir(room, n_markov=None, dtype=float):
       """Convert a Pyroomacoustics-style room.rir nested list to a MIMO tensor."""

       n_outputs = len(room.rir)
       if n_outputs == 0:
           raise ValueError("room.rir must contain at least one microphone")
       n_inputs = len(room.rir[0])
       if n_inputs == 0:
           raise ValueError("room.rir[0] must contain at least one source")

       lengths = [len(room.rir[y][u]) for y in range(n_outputs) for u in range(n_inputs)]
       n_taps = max(lengths) if n_markov is None else int(n_markov)
       markov = np.zeros((n_taps, n_outputs, n_inputs), dtype=dtype)

       for y in range(n_outputs):
           if len(room.rir[y]) != n_inputs:
               raise ValueError("all microphones must have the same number of source RIRs")
           for u in range(n_inputs):
               h = np.asarray(room.rir[y][u], dtype=dtype)
               n = min(n_taps, h.size)
               markov[:n, y, u] = h[:n]
       return markov

Then use the MIMO reducer on the resulting tensor:

.. code-block:: python

   markov = markov_from_pyroom_rir(room, n_markov=512)
   result = ld.finite_hankel_reduce_mimo(
       markov,
       reduced_order=12,
       block_rows=32,
       block_cols=32,
   )

This is a finite numerical model-reduction workflow for simulated acoustic
paths.  It does not replace Pyroomacoustics, and it is not a production
acoustic-echo-cancellation stack.

See :doc:`examples/generated/pyroomacoustics_mimo_rir_recipe` for a runnable
version using a fake ``room.rir`` object, so the recipe can be tested without
installing Pyroomacoustics.

Tangential Schur/Pick data import
---------------------------------

Tangential data often arrive from scripts as CSV, JSON, or ``.mat`` files.  The
safe import rule is: load points, directions, and values separately, coerce them
to complex NumPy arrays, and validate with ``RightTangentialSchurData`` before
building a Pick matrix.

.. code-block:: python

   import json
   import numpy as np
   import lattice_dsp as ld

   def complex_array(items):
       # JSON representation: [[real, imag], ...] or nested lists of that form.
       arr = np.asarray(items, dtype=float)
       return arr[..., 0] + 1j * arr[..., 1]

   with open("tangential_data.json", encoding="utf-8") as f:
       raw = json.load(f)

   data = ld.RightTangentialSchurData(
       complex_array(raw["points"]),
       complex_array(raw["directions"]),
       complex_array(raw["values"]),
   )
   eig = ld.pick_matrix_eigenvalues(ld.right_tangential_pick_matrix(data))

This does not synthesize a full general Schur solution.  It imports and checks
finite right-tangential data in the package's supported diagnostic subset.

WAV, eSpeak/eSpeak NG, librosa, and soundfile recipes
-----------------------------------------------------

eSpeak/eSpeak NG can be useful as a deterministic speech source for examples.
Keep it outside the Python package and generate a WAV file from the shell:

.. code-block:: bash

   espeak-ng -w speech.wav "This is a lattice DSP test signal."

or, on systems that provide the older command name:

.. code-block:: bash

   espeak -w speech.wav "This is a lattice DSP test signal."

After that, load the WAV file in your own script and pass the samples to
``lattice-dsp``.  The standard library is enough for simple 16-bit PCM WAV files:

.. code-block:: python

   import wave
   import numpy as np

   def read_mono_pcm16(path):
       with wave.open(path, "rb") as wf:
           if wf.getsampwidth() != 2:
               raise ValueError("expected 16-bit PCM WAV")
           sample_rate = wf.getframerate()
           channels = wf.getnchannels()
           raw = wf.readframes(wf.getnframes())
       x = np.frombuffer(raw, dtype="<i2").astype(float) / 32768.0
       if channels > 1:
           x = x.reshape(-1, channels).mean(axis=1)
       return sample_rate, x

   sample_rate, x = read_mono_pcm16("speech.wav")

For convenience, :doc:`examples/generated/external_audio_wav_recipe` includes a
complete no-extra-dependency WAV read/write example.  Use librosa, soundfile, or
``scipy.io.wavfile`` in your own scripts when you already depend on those
libraries:

.. code-block:: python

   import librosa
   import lattice_dsp as ld

   x, sample_rate = librosa.load("speech.wav", sr=16000, mono=True)
   y = ld.LatticeIIR([0.35, -0.2], [0.5, 0.1, 0.2]).process(x)

.. code-block:: python

   import soundfile as sf
   import lattice_dsp as ld

   x, sample_rate = sf.read("speech.wav", always_2d=False)
   y = ld.LatticeIIR([0.35, -0.2], [0.5, 0.1, 0.2]).process(x)

Scope boundary
--------------

Interoperability support is deliberately recipe-based:

* no required SciPy, python-control, MATLAB engine, or Octave dependency;
* no required Pyroomacoustics dependency;
* no required eSpeak/eSpeak NG or ``libespeak`` dependency;
* no required librosa, soundfile, or audio-codec dependency;
* no production claim for room simulation, speech synthesis, acoustic echo
  cancellation, or control-system design.

This keeps the package focused while still making it easy to bring external
signals, state-space systems, impulse responses, MIMO room paths, AR data, and
Schur/Pick interpolation data into the stable-IIR and model-reduction workflows.
