"""Generate tutorial-style Sphinx pages for examples and benchmarks.

The generated pages are intentionally not committed.  They are build artifacts
that combine narrative context, equations, captured command output, figures/data
files, and a literal include of the source script.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

try:
    from tools.benchmark_visuals import create_benchmark_visuals
except ImportError:  # pragma: no cover - supports direct script execution from tools/
    from benchmark_visuals import create_benchmark_visuals

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".webp"}
DATA_SUFFIXES = {".csv", ".json", ".txt", ".md", ".npy", ".npz"}


@dataclass(frozen=True)
class Tutorial:
    script: str
    title: str
    group: str
    purpose: str
    context: str
    equations: str
    readout: str
    sample_output: str = ""
    runtime_status: str = ""


@dataclass(frozen=True)
class BenchmarkTutorial:
    slug: str
    script: str
    title: str
    purpose: str
    context: str
    equations: str
    readout: str
    args: tuple[str, ...]


EXAMPLES: tuple[Tutorial, ...] = (
    Tutorial(
        "algorithm_selection_demo.py",
        "Choosing the right lattice-dsp algorithm",
        "Orientation tutorials",
        "Map common DSP tasks to the package APIs, diagnostics, and validation scope.",
        """
        New users often know the problem they want to solve before they know the package
        vocabulary.  This tutorial gives a compact decision table for stable IIR filtering,
        adaptive identification, AR estimation, finite-Hankel reduction, Nehari/AAK-style
        finite diagnostics, MIMO reduction, and matrix-lattice experiments.
        """,
        r"""
        The organizing principle is to choose the coordinate system that matches the
        constraint: reflection coefficients for scalar stability, AR recursions for
        prediction, Hankel singular values for input-output memory, and state-space Markov
        parameters for MIMO reduction.
        """,
        "Use the printed table as a routing guide, then follow the linked examples for the selected algorithm family.",
    ),
    Tutorial(
        "pyroomacoustics_mimo_rir_recipe.py",
        "Pyroomacoustics MIMO RIR interoperability recipe",
        "Interoperability recipes",
        "Convert Pyroomacoustics-style room impulse responses to the MIMO Markov tensor shape used by lattice-dsp.",
        """
        Room-acoustics simulators can produce one impulse response for each
        microphone/source pair.  This recipe keeps Pyroomacoustics outside the
        package dependency tree while documenting the data convention used to
        bring those paths into MIMO block-Hankel reduction.
        """,
        r"""
        The mapping is

        .. code-block:: text

           room.rir[microphone][source][tap]
                -> markov[tap, microphone, source]
        """,
        "Use the printed shape and relative Markov-response error to verify the conversion and reduced MIMO model.",
    ),
    Tutorial(
        "external_audio_wav_recipe.py",
        "External WAV and eSpeak/eSpeak NG interoperability recipe",
        "Interoperability recipes",
        "Bring an external WAV signal into lattice-dsp without adding audio I/O dependencies to the package.",
        """
        Speech synthesizers, recording tools, DAWs, and simulators can all write
        WAV files.  This recipe uses the Python standard library for a minimal
        PCM WAV boundary and leaves richer loaders such as librosa, soundfile,
        or scipy.io.wavfile as user-side choices.
        """,
        r"""
        The boundary is intentionally simple:

        .. code-block:: text

           external tool -> WAV file -> NumPy array -> lattice-dsp filter/model
        """,
        "The printed RMS values confirm that the signal crossed the WAV-to-array boundary and was processed by a lattice filter.",
    ),
    Tutorial(
        "reflection_conversion.py",
        "Reflection coefficients and denominator coefficients",
        "Core scalar lattice tutorials",
        "Convert between reflection/PARCOR coefficients and a conventional IIR denominator.",
        """
        This is the shortest path into the package.  A stable all-pole IIR filter can be
        represented either by denominator coefficients or by reflection coefficients.  The
        reflection form is more convenient for adaptive work because stability is controlled by
        simple per-stage bounds.
        """,
        r"""
        For an all-pole denominator

        .. math::

           A(z)=1+a_1z^{-1}+\cdots+a_pz^{-p},

        scalar lattice stability is guaranteed when every reflection coefficient satisfies

        .. math::

           |k_i| < 1.
        """,
        "Check that converting reflection coefficients to a denominator and back returns the original values up to numerical precision.",
    ),
    Tutorial(
        "lattice_ladder_realization.py",
        "Lattice-ladder realization versus direct numerator coefficients",
        "Core scalar lattice tutorials",
        "Show that lattice-ladder taps can realize the same transfer function as direct numerator coefficients.",
        """
        A lattice filter gives a stable recursive denominator.  Ladder taps then combine the
        internal lattice states to realize a numerator.  This example is a sanity check that the
        lattice-ladder representation and a direct numerator representation agree sample-by-sample.
        """,
        r"""
        The transfer function is still

        .. math::

           H(z)=\frac{B(z)}{A(z)},

        but ``A`` is controlled by reflection coefficients while ``B`` is represented by ladder
        taps.
        """,
        "The important number is the maximum absolute output difference; it should be close to floating-point roundoff.",
    ),
    Tutorial(
        "reflection_coefficients_stability_demo.py",
        "Reflection coefficients as a stability coordinate system",
        "Core scalar lattice tutorials",
        "Show why bounded scalar reflection/PARCOR coefficients are a practical stability coordinate system.",
        """
        Scalar IIR stability is hard to read from direct denominator coefficients.  In
        lattice coordinates, the all-pole stability condition is exposed stage by stage.
        This tutorial compares reflection-parameterized denominators with a few direct
        denominator examples near the unit-circle boundary.
        """,
        r"""
        For a scalar all-pole denominator built by the Schur step-up recursion,

        .. math::

           |k_i| < 1 \quad \text{for every stage}

        keeps the poles inside the unit disk.  Direct denominator coefficients do not
        provide an equivalent per-coefficient test.
        """,
        "Compare the maximum reflection magnitude with the maximum pole radius.  The direct denominator rows show why pole or reflection diagnostics are still needed outside lattice coordinates.",
    ),
    Tutorial(
        "stability_vs_direct_iir.py",
        "Why direct-form adaptive IIR updates can become unstable",
        "Core scalar lattice tutorials",
        "Compare an unconstrained denominator update with bounded reflection-parameterized adaptation.",
        """
        FIR LMS already has a real learning-rate problem: the step size controls convergence,
        misadjustment, and divergence.  Adaptive IIR keeps that tuning issue and adds a structural
        one, because denominator coefficients move the poles.  This tutorial intentionally uses a
        simple aggressive direct-form update to show the failure mode, then compares it with an
        adaptive lattice model whose reflection coefficients remain bounded.
        """,
        r"""
        FIR LMS uses

        .. math::

           w[n+1] = w[n] + \mu x_n e[n],

        so step-size selection is already part of the design.  Direct-form IIR adaptation also
        requires all poles of ``A(z)`` to stay inside the unit circle.  In reflection form, the
        scalar sufficient condition is simply

        .. math::

           |k_i| < 1-\epsilon.
        """,
        "Look at the pole-radius figure.  Values above 1 indicate instability in the direct denominator update; the lattice path enforces stability by construction.",
    ),
    Tutorial(
        "openmp_batch_processing.py",
        "OpenMP batch processing for independent streams",
        "Core scalar lattice tutorials",
        "Run many independent signals through the C++ backend and compare batch behavior.",
        """
        The C++ extension is most useful when the same lattice operation is applied to many
        independent channels, trials, or parameter settings.  This example demonstrates the batch
        interface used by the benchmarks.
        """,
        r"""
        If ``C`` independent channels each contain ``N`` samples, the batch path parallelizes over
        independent jobs while preserving each stream's state recursion.
        """,
        "Check whether OpenMP is reported as available and compare the batch output against the scalar reference path.",
    ),
    Tutorial(
        "million_sample_iir_throughput.py",
        "Million-sample IIR throughput for long acoustic-like tails",
        "Core scalar lattice tutorials",
        "Show why a compact IIR/lattice representation can process very long signals efficiently when a long decay has a low-order recursive description.",
        """
        Long acoustic paths and reverberant decays are often represented as long FIR impulse
        responses.  That representation is flexible, but a tail with hundreds of thousands of taps
        is expensive to process repeatedly, especially when the signal itself has millions of samples.
        When the dominant decay is well described by a stable recursive model, an IIR/lattice
        representation can keep the long memory implicitly in a small state vector.
        """,
        r"""
        A long FIR tail computes

        .. math::

           y[n] = \sum_{m=0}^{L-1} h[m] x[n-m].

        For the exponential tail

        .. math::

           h[m] = (1-r) r^m, \qquad 0 < r < 1,

        an equivalent stable IIR recursion is

        .. math::

           y[n] = (1-r) x[n] + r y[n-1].

        In the scalar lattice convention, this denominator has reflection coefficient
        ``k_1 = -r``, so stability is exposed by ``|k_1| < 1``.
        """,
        "Compare the IIR recursive state count with the FIR truncation length and the local median timing on million-sample inputs.",
    ),
    Tutorial(
        "large_order_echo_stress.py",
        "Large echo-scale recursive model stress",
        "Core scalar lattice tutorials",
        "Stress a million-sample signal with a high-order stable lattice-ladder model and compare the scale with a long FIR echo tap vector.",
        """
        Echo-style paths often have long memory: delay, early reflections, and a slowly decaying
        room tail.  FIR adaptive echo cancellers model that memory by adding taps, so a long path
        becomes a large parameter vector that is filtered and updated at every sample.  This example
        keeps adaptation out of scope and instead stresses the fixed-filter processing axis: a
        million-sample input and a high-order stable recursive lattice-ladder model.
        """,
        r"""
        A direct FIR echo model with ``L`` taps has the filtering relation

        .. math::

           y[n] = \sum_{m=0}^{L-1} h[m] x[n-m],

        and an LMS-style update touches the same large tap vector again,

        .. math::

           h_{n+1} = h_n + \mu e[n] x_n.

        A lattice-ladder IIR model stores recursive state and stage parameters.  Its scalar
        all-pole stability guard is still expressed through bounded reflection coefficients,

        .. math::

           |k_i| < 1.

        The comparison is a scale diagnostic: ``N L`` direct FIR tap visits versus ``N p``
        lattice-stage visits for recursive order ``p``.  It is not an accuracy-equivalence claim.
        """,
        "Compare the local lattice-ladder timing with the printed FIR echo-scale tap-visit estimates, especially the FIR taps / lattice order ratio.",
    ),
    Tutorial(
        "mimo_long_signal_state_space_stress.py",
        "MIMO long-signal state-space stress",
        "Model-reduction tutorials",
        "Reduce a coupled MIMO state-space model with the finite block-Hankel workflow, then process long batched multichannel signals through the compiled C++ runtime.",
        """
        This is the multichannel counterpart to the scalar long-signal stress examples.  A
        coupled MIMO system is converted to Markov parameters, reduced with
        ``finite_hankel_reduce_mimo``, and then reused on a long batched input through
        ``mimo_state_space_process_batch``.  The printed comparison numbers are scale
        diagnostics for MIMO echo-style paths, not claims of accuracy equivalence to every
        long FIR model and not a matrix-valued AAK/Nehari solver claim.
        """,
        r"""
        A MIMO state-space model uses

        .. math::

           x_s[n+1] = A x_s[n] + B u[n], \qquad
           y[n] = C x_s[n] + D u[n].

        Its Markov matrices ``M_k`` map input channels to output channels at lag ``k``.
        The finite MIMO block-Hankel reducer builds a matrix whose blocks are these
        Markov matrices and returns a lower-order state-space realization.  For comparison,
        a direct MIMO FIR echo model with ``L`` taps per input-output path has scale

        .. math::

           N \, L \, m \, p,

        for ``N`` samples, ``m`` inputs, and ``p`` outputs.  The state-space runtime has a
        dense recursive scale tied to the chosen state order instead of the FIR tap count.
        """,
        "Inspect the finite block-Hankel reduction time, retained energy, reduced runtime, output-channel throughput, and the printed direct MIMO FIR tap-visit scale.",
        r"""
        The following output is from one local run of the default 8-by-8 stress command.
        Exact timings are machine-dependent, but the scale relationship is the point:
        the finite MIMO reduction produced a stable order-16 model that processed one
        million multichannel samples while the equivalent direct MIMO FIR tap-visit
        count was orders of magnitude larger.

        .. code-block:: text

           MIMO long-signal finite-Hankel/state-space stress
           ================================================================
           batch streams: 1
           samples per stream: 1,000,000
           inputs x outputs: 8 x 8
           full MIMO state order: 64
           dominant full-model pole radius target: 0.985000
           full-model spectral radius: 0.985000
           Markov samples for reduction: 320
           block-Hankel matrix: 192 x 192
           reduced order: 16
           Markov generation time: 0.004333 s
           finite MIMO block-Hankel reduction time: 0.332755 s
           retained Hankel energy: 0.999787
           relative Markov error: 1.963e-03
           reduced model stable: True
           reduced spectral radius: 0.983021

           compiled reduced MIMO runtime
           ----------------------------------------------------------------
           median reduced state-space time: 0.151889 s
           throughput: 6.58 million multichannel samples/s
           output-channel throughput: 52.67 million output samples/s
           dense reduced state-space visits: 576,000,000
           dense visit rate: 3.79 billion visits/s
           output RMS: 1.914513

           MIMO echo-scale comparison numbers
           ----------------------------------------------------------------
           reference MIMO FIR taps per input-output path: 131,072
           FIR taps / reduced state order: 8192.0x
           full dense state-space visits at same signal size: 5,184,000,000
           reduced dense state-space visits: 576,000,000
           direct MIMO FIR filter visits: 8,388,608,000,000
           direct MIMO FIR LMS filter+update visits, rough scale: 16,777,216,000,000
           note: these are scale diagnostics, not an accuracy equivalence claim
        """,
    ),
    Tutorial(
        "adaptive_iir_system_identification.py",
        "Stable adaptive IIR system identification",
        "Adaptive and AR tutorials",
        "Identify a synthetic recursive system while keeping the learned denominator stable.",
        """
        This is a two-signal system-identification example, not one-step
        self-prediction.  A known stable target system receives the reference
        input ``x[n]`` and generates the desired signal ``d[n]``.  The adaptive
        lattice-ladder model receives the same reference input and learns a
        stable recursive approximation to the target input-output map.
        """,
        r"""
        The data relationship is

        .. math::

           d[n] = H_{\mathrm{target}}(q^{-1})x[n],
           \qquad
           \widehat d[n] = H_{\theta_n}(q^{-1})x[n].

        The instantaneous error is

        .. math::

           e[n] = d[n] - \widehat d[n].

        The adaptive model updates numerator/ladder parameters and reflection
        coefficients after forming the current output and error.  The reflection
        update is bounded so the learned denominator remains stable during
        training.  This is causal adaptive filtering because ``\widehat d[n]``
        uses current/past reference samples and previous filter state, not future
        desired samples.
        """,
        "Compare the initial and final MSE.  Also inspect the learned reflection coefficients and pole radius if printed.",
        runtime_status="""This is inductive/streaming system identification on a synthetic sequence: the target generates ``d[n]`` from the same reference ``x[n]`` seen by the adaptive filter.  It is not a single-signal predictor, and it is not a production echo canceller.""",
    ),
    Tutorial(
        "tracking_drifting_iir_system.py",
        "Tracking a drifting stable IIR system",
        "Adaptive and AR tutorials",
        "Track a target system whose parameters slowly change over time.",
        """
        Real adaptive problems are not always stationary.  This tutorial changes the target system
        gradually and checks whether bounded reflection adaptation can track the drift without
        crossing the stability boundary.
        """,
        r"""
        A useful diagnostic is the moving error power

        .. math::

           \operatorname{MSE}_t = \frac{1}{W}\sum_{i=t-W+1}^{t} e_i^2.
        """,
        "The figure should show the tracking error over time.  Slow drift should be followed; abrupt or very fast drift would require different tuning.",
    ),
    Tutorial(
        "adaptive_batch_processing.py",
        "Batched adaptive lattice trials",
        "Adaptive and AR tutorials",
        "Run independent adaptive system-identification trials through the batch API.",
        """
        Parameter sweeps and Monte Carlo trials are easier when many independent filters can be run
        together.  This example mirrors the scalar adaptive API but uses a batch-oriented entry
        point.
        """,
        r"""
        Each batch member has its own input, desired signal, and final state.  The jobs are
        independent, so they can be parallelized safely.
        """,
        "Confirm that each batch member converges and that the reported batch dimensions match the requested number of trials.",
    ),
    Tutorial(
        "tune_reflection_update_period.py",
        "Tuning the reflection update period",
        "Adaptive and AR tutorials",
        "Explore the speed/quality tradeoff from updating reflection coefficients less frequently.",
        """
        Updating denominator parameters can be more expensive than updating numerator taps.  This
        tutorial sweeps the reflection update period and reports which periods preserve quality while
        reducing work.
        """,
        r"""
        A period ``P`` updates reflection coefficients only when

        .. math::

           n \equiv 0 \pmod P.
        """,
        "Look for the largest period that keeps the tail MSE close to the period-1 baseline.",
    ),
    Tutorial(
        "adaptive_notch_tracking.py",
        "Adaptive notch tracking",
        "Applications and signal-model tutorials",
        "Track and suppress a sinusoidal interferer with a stable second-order notch model.",
        """
        Notch filters are a compact way to remove narrowband interference.  This tutorial uses a
        small adaptive example to show how a stable recursive structure can follow an interfering
        tone.
        """,
        r"""
        A second-order notch has a pair of zeros near the interference frequency and stable poles
        whose radius controls bandwidth.
        """,
        "Inspect the estimated notch frequency and the before/after error or suppression metric.",
    ),
    Tutorial(
        "adaptive_prediction_ar.py",
        "Adaptive one-step AR prediction",
        "Applications and signal-model tutorials",
        "Use a stable adaptive recursive model for one-step signal prediction.",
        """
        AR prediction is the scalar setting where lattice filters are historically very natural.
        The model predicts the current sample from past samples while preserving a stable recursive
        parameterization.
        """,
        r"""
        A one-step predictor estimates

        .. math::

           \hat{x}[n] = g(x[n-1], x[n-2], \ldots),\qquad e[n]=x[n]-\hat{x}[n].
        """,
        "The final prediction error should be smaller than the initial error after the predictor adapts.",
    ),
    Tutorial(
        "burg_levinson_ar_tools.py",
        "Burg and Levinson-Durbin AR tools",
        "Applications and signal-model tutorials",
        "Estimate AR coefficients using autocorrelation/Levinson and Burg-style recursions.",
        """
        This tutorial compares two common AR estimation routes.  Levinson-Durbin solves the
        Yule-Walker equations from autocorrelations.  Burg estimates reflection coefficients from
        forward/backward prediction errors.
        """,
        r"""
        The AR model is

        .. math::

           x[n]+a_1x[n-1]+\cdots+a_px[n-p]=e[n].
        """,
        "Compare the estimated coefficients with the known synthetic model and check the final prediction error values.",
    ),
    Tutorial(
        "ar_spectral_estimation.py",
        "AR spectral estimation from reflection coefficients",
        "Applications and signal-model tutorials",
        "Estimate an all-pole spectrum from a stable adaptive model.",
        """
        This example turns stable AR/lattice parameters into a spectral plot.  It is useful because
        many readers understand an estimated model more quickly from its frequency response than from
        coefficient lists.
        """,
        r"""
        For an all-pole AR model, the spectral shape is proportional to

        .. math::

           S(\omega) \propto \frac{1}{|A(e^{j\omega})|^2}.
        """,
        "The learned spectrum should follow the true all-pole spectrum, especially near the dominant resonances.",
    ),
    Tutorial(
        "periodogram_vs_ar_spectrum.py",
        "Periodogram versus AR spectral estimates",
        "Spectral diagnostic tutorials",
        "Compare a windowed periodogram with Levinson and Burg AR spectra on a noisy two-tone signal.",
        """
        This tutorial introduces the visual diagnostics that make the AR tools easier to interpret.
        The periodogram is a direct Fourier estimate; AR spectra are model-based and can sharpen
        peaks, but they also depend on the chosen model order.
        """,
        r"""
        The periodogram estimates power directly from the DFT,

        .. math::

           \hat S_{\mathrm{per}}(\omega)=|\mathrm{DFT}\{w[n]x[n]\}|^2,

        while an AR spectrum uses

        .. math::

           \hat S_{\mathrm{AR}}(\omega) \propto \frac{1}{|\hat A(e^{j\omega})|^2}.
        """,
        "The vertical dotted lines mark the true tones.  Compare how broad the periodogram peaks are with the sharper AR curves, and check the CSV for numeric values.",
    ),
    Tutorial(
        "capon_spectrum_demo.py",
        "Capon/MVDR spectral estimation",
        "Spectral diagnostic tutorials",
        "Use an inverse-covariance Capon spectrum to resolve nearby tones.",
        """
        Capon/MVDR spectra are useful as a high-resolution diagnostic.  They are not a replacement
        for every periodogram or AR model, but they are visually helpful when nearby narrowband
        components are hard to separate.
        """,
        r"""
        For steering vector ``a(ω)`` and loaded covariance matrix ``R``, the Capon spectrum is

        .. math::

           \hat S_{\mathrm{Capon}}(\omega)=\frac{1}{a(\omega)^H R^{-1} a(\omega)}.
        """,
        "The main figure compares periodogram, AR, and Capon curves.  The covariance-eigenvalue plot helps diagnose whether the covariance estimate is well conditioned.",
    ),
    Tutorial(
        "spectral_diagnostics_comparison.py",
        "Spectral diagnostics comparison and tuning",
        "Spectral diagnostic tutorials",
        "Compare periodogram, AR, Burg, and Capon spectra, then vary model order/aperture.",
        """
        This is the tutorial page to read after the two focused spectral examples.  It keeps the
        signal fixed and changes the diagnostic method or its tuning parameter so the reader can see
        how visual conclusions depend on modeling choices.
        """,
        r"""
        The tunable quantities are model complexity parameters: AR order ``p`` and Capon aperture
        ``M``.  Larger values can increase resolution, but they can also amplify finite-sample noise.
        """,
        "Use the second figure as a tuning guide: peaks should align with true tones, but excessive sharpness or spurious peaks are a warning sign.",
    ),
    Tutorial(
        "hinf_lms_reproduction.py",
        "LMS through the H-infinity lens",
        "Adaptive and robust filtering tutorials",
        "Reproduce the qualitative message of Hassibi--Sayed--Kailath: LMS is not only crude least-squares descent; it also has a worst-case energy-gain interpretation.",
        """
        This flagship tutorial explains a historical surprise in adaptive filtering.  The LMS
        idea goes back to Widrow and Hoff's 1960 adaptive switching work.  For more than three
        decades, it was often introduced as the inexpensive stochastic-gradient approximation to
        least squares, while RLS was the exact least-squares recursion.  Hassibi, Sayed, and
        Kailath then showed that LMS also has a deterministic robust-filtering interpretation:
        with the right viewpoint, the algorithm is tied to an H-infinity minimax energy-gain
        problem rather than only to an average squared-error objective.

        That historical angle is useful because it changes the way readers interpret a familiar
        algorithm.  The script below does not try to reprint every derivation from the 1996 paper.
        Instead it builds a finite-horizon diagnostic that readers can inspect: for fixed
        regressors, the map from additive disturbance to prediction error is a linear operator.
        Its largest singular value exposes the disturbance direction that causes the largest
        error-energy amplification.
        """,
        r"""
        The adaptive filtering model is

        .. math::

           d_i = u_i^T w_\star + v_i,

        where ``u_i`` is the regressor, ``w_*`` is the unknown vector, and ``v_i`` is a disturbance.
        Least-squares thinking focuses on sums such as

        .. math::

           \sum_i |d_i-u_i^T\hat w_i|^2.

        The robust H-infinity diagnostic instead asks for a worst-case energy gain.  In this
        tutorial we estimate, for each algorithm,

        .. math::

           \sup_{v\ne0} \frac{\|e(v)-e(0)\|_2^2}{\|v\|_2^2},

        by forming the finite-horizon sensitivity matrix from disturbance samples to
        noise-induced prediction errors.
        """,
        "Read the first plot in the classical least-squares way: RLS converges fastest under benign random noise.  Then read the gain plots in the minimax way: the same estimator can have a larger worst-case disturbance direction.  That change of viewpoint is the lesson.",
    ),
    Tutorial(
        "rls_lattice_identification.py",
        "RLS-style lattice-ladder identification",
        "Applications and signal-model tutorials",
        "Compare RLS-style adaptation with NLMS on a small stable identification problem.",
        """
        RLS updates can converge faster than NLMS when the input is correlated, at the cost of more
        state and computation.  This example keeps the denominator stable and focuses on the adaptive
        numerator/tap behavior.
        """,
        r"""
        RLS maintains an inverse covariance estimate ``P`` and uses a gain vector that depends on the
        current regressor and forgetting factor.
        """,
        "Compare the final errors and convergence behavior for the RLS and NLMS paths.",
    ),
    Tutorial(
        "streaming_block_processing.py",
        "Streaming block processing equivalence",
        "Applications and signal-model tutorials",
        "Show that stateful block processing matches one-shot processing.",
        """
        Real-time DSP usually processes blocks rather than entire arrays.  This tutorial checks that
        block boundaries do not change the result when filter state is carried correctly.
        """,
        r"""
        For a stateful recursion, the output over concatenated blocks should match the output over
        the full signal when final state from one block is used as initial state for the next.
        """,
        "The reported maximum difference should be close to roundoff.",
    ),
    Tutorial(
        "multichannel_levinson_ar.py",
        "Multichannel AR with block Levinson-Durbin",
        "Multichannel and matrix tutorials",
        "Estimate a vector AR model and compare block Levinson against a dense block-Toeplitz solve.",
        """
        Scalar AR models generalize to vector autoregressive models where each lag coefficient is a
        matrix.  This tutorial shows how block Toeplitz structure and matrix reflection coefficients
        enter the multichannel setting.
        """,
        r"""
        Let :math:`x[n]\in\mathbb{R}^c` be a vector signal and let
        :math:`e[n]` be the prediction residual.  An order-``p`` vector AR model is

        .. math::

           x[n] + \sum_{k=1}^{p} A_k x[n-k] = e[n],
           \qquad A_k \in \mathbb{R}^{c\times c}.

        The sample autocovariances :math:`R_\ell=\mathbb{E}\{x[n]x[n-\ell]^T\}`
        form a block-Toeplitz Yule--Walker system for the matrices
        :math:`A_1,\ldots,A_p`.  Block Levinson--Durbin solves this system
        recursively and also exposes matrix reflection coefficients :math:`K_i`.
        The scalar condition :math:`|k_i|<1` becomes the practical matrix diagnostic
       
        .. math::

           \lVert K_i\rVert_2 < 1.

        The coefficient heatmaps show the estimated :math:`A_k`; the reflection plot
        checks the stage norms; the residual-covariance plot checks the remaining
        multichannel prediction error.
        """,
        "Check the coefficient difference against the direct solve, then use the coefficient heatmaps, reflection-norm plot, and residual-covariance plot to see what the matrix AR fit learned.",
        runtime_status="""``multichannel_autocorrelation`` and ``block_levinson_durbin`` are batch estimation steps: they use a finite multichannel record or covariance sequence.  The fitted VAR recursion is causal after the matrices are known, because prediction uses only past samples ``x[n-k]`` and the current state/history.""",
    ),
    Tutorial(
        "causal_mimo_lattice_prediction.py",
        "Causal online MIMO lattice prediction",
        "Multichannel and matrix tutorials",
        "Use block-Levinson matrix reflections in a sample-by-sample causal MIMO lattice predictor.",
        """
        This tutorial separates coefficient estimation from runtime filtering.  The
        matrix reflection coefficients are estimated once from a finite training
        record, then a stateful lattice predictor runs online.  At each time step it
        predicts the next vector from stored backward-error states before the current
        vector is observed.
        """,
        r"""
        With forward and backward matrix reflection coefficients :math:`K_m` and
        :math:`L_m`, the online vector lattice recursion is

        .. math::

           f_0[n] = b_0[n] = y[n],

        .. math::

           f_m[n] = f_{m-1}[n] + K_m b_{m-1}[n-1],

        .. math::

           b_m[n] = b_{m-1}[n-1] + L_m f_{m-1}[n].

        The one-step prediction is obtained before seeing :math:`y[n]` by evaluating
        the same recursion with :math:`y[n]=0` and negating the result:

        .. math::

           \hat y[n] = -f_p[n]\big|_{y[n]=0}.

        After the true vector is observed, :math:`f_p[n]=y[n]-\hat y[n]` is the
        forward prediction error and the backward-error states are updated.  This is
        causal in the prediction sense: :math:`\hat y[n]` depends only on stored
        states from samples :math:`< n`.
        """,
        "Check that the online lattice residual matches the direct AR residual from the same block-Levinson fit, then inspect the trace, reflection-norm, and residual-covariance figures.",
        runtime_status="""The block-Levinson fit is a batch estimation step.  The ``MIMOLatticePredictor`` object created from those reflection matrices is a runtime object: ``predict()`` uses only previous vectors, and ``update(y_n)`` consumes the current vector afterward.""",
    ),
    Tutorial(
        "matrix_ar_spectral_estimation.py",
        "Matrix-valued AR spectral estimation",
        "Multichannel and matrix tutorials",
        "Compute a multichannel AR frequency response and spectral-matrix diagnostic.",
        """
        Multichannel spectra are matrices, not scalar curves.  This example evaluates matrix-valued
        AR responses so channel interactions and cross-spectral behavior can be inspected.
        """,
        r"""
        For the matrix AR model, define the polynomial matrix

        .. math::

           A(z) = I + \sum_{k=1}^{p} A_k z^{-k}.

        The transfer matrix from innovation :math:`e[n]` to signal :math:`x[n]` is

        .. math::

           H(e^{j\omega}) = A(e^{j\omega})^{-1}.

        If the innovation covariance is :math:`\Sigma_e`, the spectral-density
        matrix is

        .. math::

           S_x(e^{j\omega})
             = H(e^{j\omega})\,\Sigma_e\,H(e^{j\omega})^H.

        Diagonal entries of :math:`S_x` are auto-spectra.  Off-diagonal entries are
        cross-spectra, so their magnitude and phase indicate frequency-dependent
        coupling between channels.
        """,
        "Use the auto-spectrum and cross-spectrum figures to see both per-channel power and cross-channel coupling, then check the spectral radius and reflection norms for stability.",
        runtime_status="""The spectral-density calculation is an offline frequency-grid diagnostic of an already fitted VAR model.  If the fitted model is stable, the corresponding time-domain VAR/IIR recursion is causal; the plot itself is not a streaming operation.""",
    ),
    Tutorial(
        "mimo_lattice_vs_block_levinson.py",
        "MIMO lattice response versus block Levinson AR",
        "Multichannel and matrix tutorials",
        "Compare two matrix-valued views: all-pass lattice filtering and vector AR estimation.",
        """
        This tutorial puts MIMO/matrix lattice objects next to multichannel AR estimation.  They are
        different models, but both rely on matrix-valued stability or reflection ideas.
        """,
        r"""
        The block-Levinson side fits a predictive VAR model

        .. math::

           A(z)=I+\sum_{k=1}^{p}A_k z^{-k},
           \qquad H_{AR}(z)=A(z)^{-1},

        whose stability is checked through the companion matrix :math:`C_A`:

        .. math::

           \rho(C_A) < 1.

        The matrix-lattice side uses reflection matrices :math:`K_i` with
        :math:`\lVert K_i\rVert_2<1` to build a frequency response :math:`G(z)`.
        In this all-pass diagnostic example, the desired property is

        .. math::

           G(e^{j\omega})^H G(e^{j\omega}) \approx I.

        The point of the example is not to claim that the two constructions are the
        same model.  It places their diagnostics side by side: VAR prediction error
        and companion stability versus lattice reflection norms and frequency-wise
        unitarity.
        """,
        "Use the reflection-norm and singular-value figures to separate the two diagnostics: block Levinson validates VAR prediction, while the matrix lattice validates frequency-wise unitarity.",
        runtime_status="""The block-Levinson side is batch coefficient estimation from covariance data.  The matrix-lattice side evaluates a response on a frequency grid.  This page compares diagnostics; it is not a sample-by-sample runtime filter.""",
    ),
    Tutorial(
        "mimo_diagonal_equals_independent_siso.py",
        "Diagonal MIMO equals independent SISO",
        "Multichannel and matrix tutorials",
        "Use independent SISO lattice filters and online predictors as sanity checks for diagonal MIMO.",
        """
        Before studying coupled MIMO lattice systems, it helps to inspect the diagonal case.  If
        every matrix Markov parameter or matrix reflection coefficient is diagonal, no output
        channel depends on any other input channel.  The MIMO system is exactly several scalar SISO
        systems running side by side.

        This is the intuition bridge between scalar lattice filters and matrix-valued MIMO filters:
        scalar reflection coefficients become diagonal entries first, and only later become
        genuinely coupled matrices.
        """,
        r"""
        A diagonal MIMO impulse response has Markov matrices

        .. math::

           M_k = \operatorname{diag}(h^{(1)}_k, \ldots, h^{(p)}_k),

        and therefore a diagonal transfer matrix

        .. math::

           H(z)=\operatorname{diag}\bigl(H_1(z),\ldots,H_p(z)\bigr).

        The convolution separates by channel:

        .. math::

           y_i[n] = \sum_k h^{(i)}_k x_i[n-k],
           \qquad y_i \text{ does not depend on } x_j \text{ for } j\ne i.

        The same diagonal reduction should hold for the online lattice predictor.  If
        the matrix reflection coefficients are diagonal,

        .. math::

           K_m = \operatorname{diag}(k_{m,1},\ldots,k_{m,p}),
           \qquad
           L_m = \operatorname{diag}(\ell_{m,1},\ldots,\ell_{m,p}),

        then the vector recursion decouples into independent one-channel recursions:

        .. math::

           \widehat y[n]
           = \begin{bmatrix}
              \widehat y_1[n] & \cdots & \widehat y_p[n]
             \end{bmatrix}^T.

        This is the algebraic reason the diagonal MIMO result must match running scalar
        SISO filters or one-channel online lattice predictors independently.  Any nonzero
        off-diagonal Markov block or reflection entry would create true channel coupling.
        """,
        "The MIMO/SISO differences should be near numerical precision for both the finite impulse-response comparison and the online predict-before-update comparison.",
        runtime_status="""The scalar ``LatticeIIR`` filters and the online ``MIMOLatticePredictor`` comparison are causal.  The Markov-convolution part is evaluated on a stored array as a validation check, but each output sample uses only lags ``x[n-k]``.  The online predictor part explicitly calls ``predict()`` before ``update(y_n)`` and therefore uses only previous samples when forming ``\\widehat y[n]``.""",
    ),
    Tutorial(
        "online_coupled_mimo_vs_siso.py",
        "Online coupled MIMO prediction versus independent SISO",
        "Multichannel and matrix tutorials",
        "Show that full online MIMO lattice prediction captures cross-channel dynamics that independent SISO predictors miss.",
        """
        The diagonal-MIMO tutorial is a reduction test: when reflection matrices are
        diagonal, MIMO equals independent SISO.  This tutorial tests the complementary
        case.  A synthetic vector AR process contains off-diagonal lag matrices, so
        previous samples from one channel help predict another channel.  A full MIMO
        lattice predictor can use that cross-channel history; per-channel SISO
        predictors cannot.
        """,
        r"""
        The training signal follows a coupled vector AR model

        .. math::

           y[n] + \sum_{k=1}^{p} A_k y[n-k] = e[n],
           \qquad A_k \in \mathbb{R}^{c\times c}.

        Off-diagonal entries of :math:`A_k` encode cross-channel dynamics.  The full
        MIMO predictor estimates matrix reflection coefficients and predicts

        .. math::

           \widehat y[n] = g\bigl(y[n-1], y[n-2], \ldots\bigr),

        where :math:`g` may mix channels through matrix coefficients.  Independent
        SISO baselines instead fit one predictor per channel,

        .. math::

           \widehat y_i[n] = g_i\bigl(y_i[n-1], y_i[n-2], \ldots\bigr),

        so they cannot use :math:`y_j[n-k]` for :math:`j\ne i`.  The example also
        includes a diagonal ablation of the full MIMO reflection matrices to show what
        happens when the learned off-diagonal entries are removed.
        """,
        "Compare residual RMS and residual-covariance plots.  The full MIMO residual should be smaller and less cross-correlated than the independent SISO baseline when the data really contain cross-channel dynamics.  The off-diagonal residual-correlation reduction is often the clearer MIMO diagnostic than scalar RMS alone.",
        runtime_status=r"""The coefficients are estimated from a finite training record.  Prediction on the held-out test record is online: each model calls ``predict()`` before ``update(y_n)``, so ``\widehat y[n]`` uses only previous test vectors or previous samples from the corresponding SISO channel.""",
    ),
    Tutorial(
        "reachability_observability_hankel_demo.py",
        "Reachability, observability, and Hankel singular values",
        "Model-reduction tutorials",
        "Connect state-space reachability and observability with finite Hankel singular values.",
        """
        Hankel singular values are easier to interpret once they are tied to state-space
        reachability and observability.  This tutorial builds a small system with
        unreachable and unobservable state directions and shows that the input-output
        Hankel matrix only captures directions that are both excited by inputs and seen
        at outputs.
        """,
        r"""
        For a state-space model ``(A, B, C, D)``, Markov parameters satisfy

        .. math::

           M_k = C A^{k-1} B,

        and a block Hankel matrix factors as

        .. math::

           \mathcal H = \mathcal O\,\mathcal R,

        where ``R`` is reachability and ``O`` is observability.
        """,
        "The reachability and observability ranks are both three in the toy model, but the finite Hankel matrix has only two significant singular values because only two directions are both reachable and observable.",
    ),
    Tutorial(
        "finite_hankel_model_reduction.py",
        "Finite Hankel SISO model reduction",
        "Model-reduction tutorials",
        "Build a finite Hankel matrix from a stable IIR impulse response and construct lower-order rational models with the C++ backend.",
        """
        This tutorial is the first executable bridge between the model-reduction theory page and
        the package API.  It is deliberately finite-dimensional: a truncated Hankel matrix is built
        from the impulse response, its singular values are inspected, and a lower-order Ho--Kalman
        realization is recovered from the leading Hankel factors.

        Exact AAK/Nehari theory is an infinite-dimensional optimal rational-approximation theory.
        The implementation here is a practical finite-section reducer inspired by the same
        Hankel-operator viewpoint, but it does not claim exact Nehari or AAK optimality.
        """,
        r"""
        Given an impulse response ``h[n]``, the finite Hankel matrix is

        .. math::

           H_{ij}=h[i+j+1].

        Its singular values measure input-output memory.  A reduced order ``r`` keeps the leading
        singular directions and forms a balanced finite realization

        .. math::

           H_0 \approx U_r \Sigma_r V_r^T,
           \qquad
           A_r = \Sigma_r^{-1/2}U_r^T H_1 V_r\Sigma_r^{-1/2}.
        """,
        "Look for Hankel singular-value decay, retained Hankel energy, impulse-response error, and whether the reduced denominator remains stable.",
    ),
    Tutorial(
        "nehari_aak_siso_toy.py",
        "Nehari and AAK intuition from a finite SISO Hankel matrix",
        "Model-reduction tutorials",
        "Show the finite-Hankel singular-value picture behind Nehari/AAK model-reduction theory within a finite-section scope.",
        """
        The finite-Hankel reducer already available in the package is a practical baseline.  This
        tutorial explains the finite-section Nehari/AAK perspective.  It calls ``finite_nehari_approximate_tail`` to
        build a finite Hankel matrix from an anticausal tail and study its singular values.

        The goal is to make the intuition precise.  A low-rank Hankel structure means that much of
        the input-output memory is carried by a few modes.  In the finite matrix case, the next
        singular value controls the best unconstrained rank-r spectral-norm error.  AAK/Nehari is
        the rational/Hankel-structured analogue of that statement.
        """,
        r"""
        Given anticausal coefficients ``gamma_1, gamma_2, ...``, form the finite Hankel matrix

        .. math::

           \Gamma =
           \begin{bmatrix}
             \gamma_1 & \gamma_2 & \gamma_3 & \cdots \\
             \gamma_2 & \gamma_3 & \gamma_4 & \cdots \\
             \gamma_3 & \gamma_4 & \gamma_5 & \cdots \\
             \vdots & \vdots & \vdots & \ddots
           \end{bmatrix}.

        For a finite matrix, the Eckart--Young theorem gives

        .. math::

           \min_{\operatorname{rank}(X)\le r} \|\Gamma-X\|_2 = \sigma_{r+1}(\Gamma).

        The tutorial also Hankelizes the truncated SVD approximation by anti-diagonal averaging to
        show the difference between unconstrained low-rank approximation and a Hankel-structured
        approximation.
        """,
        "Check that the finite SVD error matches the next singular value, then compare it with the Hankelized approximation error.  This is a finite-section validation bridge and not an exact Nehari/AAK solver.",
    ),
    Tutorial(
        "finite_nehari_rational_bridge.py",
        "Finite Nehari tail to rational model",
        "Model-reduction tutorials",
        "Connect the finite Nehari tail approximation to a low-order recursive/rational SISO model.",
        """
        The previous Nehari/AAK toy tutorial stops at the finite Hankel matrix.  This tutorial takes
        one more conservative step: after calling ``finite_nehari_approximate_tail``, it fits a
        short linear recurrence to the Hankelized tail and realizes that recurrence as a stable
        rational/IIR impulse response.

        This is the practical bridge from Hankel singular values to recursive filters.  It still is
        not a full infinite-dimensional AAK or Nehari solver, but it shows how a low-rank Hankel
        tail can become a small rational model with poles inside the unit circle.
        """,
        r"""
        A finite-rank Hankel tail is generated by a sum of exponentials,

        .. math::

           \gamma_n \approx \sum_{i=1}^r c_i p_i^n, \qquad |p_i|<1.

        Equivalently, the tail satisfies a linear recurrence

        .. math::

           \gamma_n + a_1\gamma_{n-1}+\cdots+a_r\gamma_{n-r}\approx 0.

        The fitted recurrence gives a scalar denominator whose roots are the poles ``p_i``.
        """,
        "Look for singular-value decay, agreement between the Hankelized tail and the rational realization, and fitted poles inside the unit circle.",
    ),
    Tutorial(
        "finite_nehari_exact_rational_tail.py",
        "Exact rational-tail validation for finite Nehari candidates",
        "Model-reduction tutorials",
        "Validate the finite Nehari/rational workflow on a known rank-3 exponential tail.",
        """
        The finite workflow is validated here on a controlled exact case.  This tutorial
        constructs an anticausal tail as a known sum of three stable
        exponentials.  Such a sequence has finite Hankel rank three, so ranks one and two should
        fail the tolerance criteria while rank three should recover the tail and poles to near
        numerical precision.

        This page is a regression-style validation tutorial, not an optimality claim.  It checks
        that the package-level candidate-selection workflow behaves correctly on a case where the
        effective rational order is known in advance.
        """,
        r"""
        The exact tail is

        .. math::

           \gamma_n = \sum_{i=1}^3 c_i p_i^n, \qquad |p_i|<1.

        This implies a rank-three Hankel structure and a third-order recurrence

        .. math::

           \gamma_n + a_1\gamma_{n-1}+a_2\gamma_{n-2}+a_3\gamma_{n-3}=0.

        A correct finite rational-candidate workflow should select rank three under tight error
        and pole-radius thresholds.
        """,
        "Look for ranks 1 and 2 to be rejected, rank 3 to be selected, tiny rational error, and fitted poles matching the known stable poles.",
    ),
    Tutorial(
        "aak_siso_schmidt_pair_demo.py",
        "AAK Schmidt-pair diagnostics for SISO Hankel approximation",
        "Model-reduction tutorials",
        "Visualize the first neglected Hankel singular direction that drives the AAK/Nehari approximation barrier.",
        """
        The finite Nehari toy and rational-bridge tutorials show the singular values and a practical
        recurrence fit.  This tutorial focuses on the next AAK object: the finite Schmidt pair at
        the first neglected singular value.  In the scalar AAK theory, analogous singular-vector
        data carries the structure used to build an optimal rational approximant.

        This page is still finite-dimensional and diagnostic.  It does not expose a production
        ``aak_reduce_iir`` routine.  Instead, it shows what such a routine would need to control:
        the critical singular direction, the finite rank-r barrier, the Hankelized approximation,
        and the resulting rational poles.
        """,
        r"""
        For a finite Hankel matrix ``H`` with singular value decomposition

        .. math::

           H = U\Sigma V^T,

        the first neglected mode after a rank-``r`` approximation is the finite Schmidt pair
        ``(u_{r+1}, v_{r+1})`` satisfying

        .. math::

           H v_{r+1}=\sigma_{r+1}u_{r+1},\qquad
           H^T u_{r+1}=\sigma_{r+1}v_{r+1}.

        The scalar AAK/Nehari theory can be viewed as the infinite-dimensional rational analogue of
        this finite singular-direction picture.
        """,
        "Look for the highlighted first neglected singular value, small Schmidt-pair residuals, and how the rational fits improve as the rank crosses the critical modes.",
    ),
    Tutorial(
        "aak_siso_certificate_demo.py",
        "Finite-section SISO AAK/Nehari certificate",
        "Model-reduction tutorials",
        "Use Schmidt-pair identities to certify the finite AAK/Nehari target and attach the rational candidate.",
        """
        This is the first tutorial whose structure is deliberately AAK/Nehari-shaped.  It builds a
        finite Hankel matrix from an exact rational anticausal tail, computes the first neglected
        Schmidt pair, checks the singular-vector identities, and attaches the finite Nehari/rational
        candidate for the same rank.

        The tutorial is intentionally finite-section.  It is closer to the AAK/Nehari construction
        than the earlier rank sweeps because it verifies the Schmidt-pair equations directly, but it
        is still not advertised as a full infinite-dimensional Hardy-space solver.
        """,
        r"""
        For target rank ``r``, the finite certificate reports the first neglected singular value

        .. math::

           \sigma_{r+1}.

        It also checks the finite Schmidt-pair equations

        .. math::

           H v_{r+1}=\sigma_{r+1}u_{r+1},\qquad
           H^T u_{r+1}=\sigma_{r+1}v_{r+1}.

        On an exact rank-three rational tail, the rank-three certificate should recover the stable
        poles and produce residuals near machine precision.
        """,
        "Look for small Schmidt-pair residuals, rank-3 pole recovery, tiny rational error, and poles inside the unit disk.",
    ),
    Tutorial(
        "aak_siso_candidate_selection.py",
        "Selecting a finite SISO AAK/Nehari rational candidate",
        "Model-reduction tutorials",
        "Turn the finite Schmidt-pair and rational-bridge diagnostics into a tolerance-based candidate-selection workflow.",
        """
        The Schmidt-pair tutorial visualizes the singular direction that blocks lower-rank
        approximation.  This page uses a practical candidate workflow: choose a rank, build the finite
        Nehari Hankelized tail, fit a rational recurrence, and decide whether the candidate meets
        accuracy and stability thresholds.

        This is still a finite-dimensional candidate selector, not an exact infinite-dimensional
        AAK/Nehari solver.  The reusable logic is exposed as
        ``finite_nehari_rational_candidates`` and ``select_finite_nehari_candidate``.  Its purpose
        is to make the finite-section criteria explicit: singular-value target, Hankelized
        tail error, rational realization error, and pole radius.
        """,
        r"""
        For candidate rank ``r``, the tutorial reports

        .. math::

           \sigma_{r+1},\qquad
           \frac{\|\gamma-\widehat\gamma_r\|_2}{\|\gamma\|_2},\qquad
           \frac{\|\gamma-g_r\|_2}{\|\gamma\|_2},\qquad
           \max_i |p_i|.

        Here ``\widehat\gamma_r`` is the Hankelized finite Nehari tail, ``g_r`` is the rational
        recurrence realization, and ``p_i`` are the fitted poles.  The first rank satisfying all
        tolerances is selected.
        """,
        "Look for the first accepted rank and verify that its rational error is below tolerance while the fitted poles remain inside the unit disk.",
    ),
    Tutorial(
        "finite_aak_noisy_tail_demo.py",
        "Finite-section AAK/Nehari reduction on a non-exact tail",
        "Model-reduction tutorials",
        "Use the high-level finite AAK/Nehari reducer on a stable tail with a small non-rational residual.",
        """
        Exact rational-tail validation is necessary but too clean.  This tutorial adds a small
        deterministic residual to a stable rank-three tail and then calls the high-level
        ``finite_aak_reduce_tail`` helper.  The helper evaluates candidate ranks, selects the first
        stable rational model that meets user-chosen tolerances, and attaches a finite Schmidt-pair
        certificate for the selected rank.

        The tutorial is intentionally finite-section.  It shows how the current package behaves on
        non-exact data while keeping the scope finite-dimensional.
        """,
        r"""
        The observed tail is

        .. math::

           \gamma_n = \sum_{i=1}^3 c_i p_i^n + \epsilon_n,
           \qquad |p_i|<1,

        where ``epsilon_n`` is a small deterministic residual.  For each candidate rank ``r``, the
        helper reports

        .. math::

           \sigma_{r+1},\qquad
           \frac{\|\gamma-\widehat\gamma_r\|_2}{\|\gamma\|_2},\qquad
           \frac{\|\gamma-g_r\|_2}{\|\gamma\|_2},\qquad
           \max_i |p_i|.

        The selected candidate is the first rank satisfying the supplied error and pole-radius
        tolerances.
        """,
        "Look for a selected stable rank, a rational error below tolerance, and Schmidt-pair residuals near numerical precision for the selected finite Hankel certificate.",
    ),
    Tutorial(
        "finite_aak_iir_reduction_demo.py",
        "Finite-section AAK/Nehari reduction of a stable IIR filter",
        "Model-reduction tutorials",
        "Apply the finite AAK/Nehari tail reducer to a full stable SISO IIR filter and compare the selected reduced model.",
        """
        The previous finite AAK/Nehari pages work with abstract tail sequences.  This tutorial
        closes the loop for DSP users: start with a stable higher-order lattice/IIR filter,
        compute its impulse response, select a reduced rational model, convert the selected
        denominator back to reflection coefficients, and run the reduced filter on real signals.

        The point is practical rather than absolute optimality.  The current implementation is a
        finite-section SISO reduction candidate with Hankel, Schmidt-pair, and rational diagnostics;
        it is not a full infinite-dimensional AAK/Nehari solver.
        """,
        r"""
        A stable full model has transfer function

        .. math::

           H(z)=\frac{B(z)}{A(z)},\qquad |\lambda_i(A)|<1.

        The finite AAK/Nehari workflow uses the impulse response ``h_n`` to build a finite Hankel
        matrix, selects a rational order ``r``, and returns a reduced model

        .. math::

           H_r(z)=\frac{B_r(z)}{A_r(z)},

        whose denominator is converted back to reflection coefficients when stable.
        """,
        "Look for the selected rank, impulse and magnitude-response error, pole radius below one, and filtering speedup on a random batch.",
    ),
    Tutorial(
        "mimo_finite_hankel_model_reduction.py",
        "Finite block-Hankel MIMO model reduction",
        "Model-reduction tutorials",
        "Generalize the SISO finite-Hankel reducer to MIMO Markov matrices and return a reduced state-space model.",
        """
        The SISO reducer works with one impulse response.  A MIMO system has a sequence of Markov
        matrices instead, where each matrix maps input channels to output channels at a particular
        lag.  The natural finite-Hankel baseline is therefore a block-Hankel matrix.

        This tutorial is deliberately a baseline: it gives a reference MIMO Ho--Kalman/ERA-style
        reduction as a finite block-Hankel baseline separate from matrix Nehari or AAK optimality claims.
        """,
        r"""
        For Markov matrices ``M_k`` with shape ``outputs x inputs``, the finite block-Hankel matrix is

        .. math::

           \mathcal{H}_0 =
           \begin{bmatrix}
             M_1 & M_2 & M_3 & \cdots \\
             M_2 & M_3 & M_4 & \cdots \\
             M_3 & M_4 & M_5 & \cdots \\
             \vdots & \vdots & \vdots & \ddots
           \end{bmatrix}.

        The reduced model is returned as state-space matrices ``A, B, C, D`` rather than as scalar
        numerator/denominator coefficients.
        """,
        "Look for block-Hankel singular-value decay, retained energy, Markov-response error, and stable reduced state matrices.",
    ),
    Tutorial(
        "mimo_coupled_model_reduction.py",
        "Coupled MIMO finite-Hankel model reduction",
        "Model-reduction tutorials",
        "Reduce a genuinely coupled MIMO state-space system with the finite block-Hankel baseline.",
        """
        The diagonal MIMO tutorial shows that independent SISO filters are a special case.  This
        tutorial uses a dense, stable state-space system where each input can affect each output.
        The goal is to validate the practical MIMO finite-section baseline on a coupled
        system while keeping it separate from matrix AAK/Nehari algorithms.

        The reducer works with Markov matrices and returns a reduced state-space realization.  This
        is the natural representation for MIMO; scalar numerator/denominator coefficients are not
        forced onto a multivariable system.
        """,
        r"""
        A coupled MIMO state-space model has

        .. math::

           x_{n+1}=Ax_n+Bu_n,\qquad y_n=Cx_n+Du_n.

        Its Markov matrices are

        .. math::

           M_0=D,\qquad M_k=CA^{k-1}B\quad(k\ge 1).

        The finite block-Hankel reducer constructs a reduced model
        ``(A_r,B_r,C_r,D)`` from the leading singular directions of the block-Hankel matrix.
        """,
        "Look for nonzero off-diagonal channels, block-Hankel singular-value decay, decreasing Markov/output error with order, and reduced state radii below one.",
    ),
    Tutorial(
        "mimo_model_reduction_stress_cases.py",
        "MIMO model-reduction stress cases",
        "Model-reduction tutorials",
        "Compare finite block-Hankel MIMO reduction on three difficult matrix impulse-response families and attach tangential-Schur/Pick diagnostics.",
        """
        This tutorial collects three stress cases inspired by examples often used in
        multivariable model-reduction discussions.  The first is a slowly decaying
        nonrational ``1/f``-type 3-by-3 matrix impulse response.  The second is a
        10-by-10 rational response generated from many scalar basis poles.  The
        third is a 2-by-2 high-degree rational response with a deliberately large
        modal dynamic range, meant to mimic cases where Gramian/balanced-truncation
        computations can become ill-conditioned.

        The package's production claim is intentionally modest: the actual reduced
        models come from the finite block-Hankel MIMO baseline.  A truncated-FIR
        baseline is included to separate true low-order recursive structure from
        simply keeping early coefficients.  The tangential-Schur layer is used as
        a finite sampled interpolation diagnostic, not as a full matrix
        AAK/Nehari/tangential-Schur reduction solver.
        """,
        r"""
        Each case provides Markov matrices :math:`M_k\in\mathbb{R}^{p\times m}`.
        The finite block-Hankel matrix is

        .. math::

           \mathcal H_0 =
           \begin{bmatrix}
             M_1 & M_2 & \cdots \\
             M_2 & M_3 & \cdots \\
             \vdots & \vdots & \ddots
           \end{bmatrix}.

        The tutorial compares two explicit reduction methods.

        **Finite block-Hankel MIMO reduction.**  For a reduced state dimension
        :math:`r`, the finite Ho--Kalman/block-Hankel reducer takes the leading
        singular directions of :math:`\mathcal H_0` and returns
        :math:`(A_r,B_r,C_r,D)`.  The reduced Markov matrices are

        .. math::

           \widehat M_0^{(r)}=D,\qquad
           \widehat M_k^{(r)}=C_r A_r^{k-1}B_r \quad (k\ge1).

        **Truncated FIR baseline.**  The order-:math:`r` FIR baseline keeps the
        first :math:`r+1` Markov blocks and sets the tail to zero:

        .. math::

           \widehat M_k^{FIR,r}=\begin{cases}
             M_k, & 0\le k\le r,\\
             0, & k>r.
           \end{cases}

        Both are compared on the first :math:`N` coefficients using

        .. math::

           \frac{\left(\sum_{k=0}^{N-1}\lVert M_k-\widehat M_k^{(r)}\rVert_F^2\right)^{1/2}}
                {\left(\sum_{k=0}^{N-1}\lVert M_k\rVert_F^2\right)^{1/2}}.

        The second error figure reports a finite-Hankel spectral-norm diagnostic:
        the block-Hankel method uses :math:`\sigma_{r+1}(\mathcal H_0)/\sigma_1(\mathcal H_0)`,
        while the FIR baseline uses the normalized spectral norm of the
        block-Hankel matrix built from the FIR truncation error.

        The tangential-Schur diagnostic samples points :math:`z_i` in the unit disk
        and right directions :math:`u_i`, then checks scaled data

        .. math::

           S(z_i)u_i = F(z_i)u_i/\gamma

        with the Pick matrix

        .. math::

           P_{ij}=\frac{u_i^*u_j-v_i^*v_j}{1-\overline z_i z_j}.

        Increasing :math:`\gamma` until :math:`P\succeq0` gives a finite sampled
        Schur-feasibility scale.  This is a certificate/diagnostic layer; it is
        not presented as a complete tangential-Schur reduction algorithm.
        """,
        "Compare H2/Markov error, finite-Hankel spectral-norm tail error, timing curves, block-Hankel singular-value decay, scaled Pick eigenvalues, and the conditioning note for the high-degree case.  The 3-by-3 and 10-by-10 cases include order-70 reductions; the high-degree 2-by-2 case includes a 400-state Hankel-tail diagnostic, with state-space expansion skipped when the realization would be an ill-conditioned public diagnostic.",
        runtime_status="""All three cases are offline model-reduction diagnostics.  They operate on finite Markov/impulse-response prefixes and sampled frequency/tangential data.  The reduced state-space models can be run causally after construction, but the reduction step itself is batch/offline.""",
    ),
    Tutorial(
        "mimo_hankel_to_matrix_lattice_bridge.py",
        "MIMO block-Hankel to matrix-lattice bridge diagnostics",
        "Model-reduction tutorials",
        "Use reduced MIMO Markov data to seed a stable matrix-lattice all-pass scaffold and measure the realization gap.",
        """
        This page defines the bridge scope between the MIMO block-Hankel reducer and the
        matrix-lattice direction.  A general reduced MIMO state-space model has frequency-dependent
        gains, while a matrix-lattice all-pass is unitary.  The example therefore compares a stable
        lattice scaffold with the reduced model's unitary polar factor as a diagnostic, not an exact
        realization solver.
        """,
        r"""
        For a reduced response ``H(e^{j\omega})``, the polar factor is the unitary matrix

        .. math::

           U_p(e^{j\omega}) = U(e^{j\omega})V(e^{j\omega})^H,

        where ``H=U\Sigma V^H``.  The scaffold error reports how close a finite matrix-lattice
        all-pass response is to this unitary part.
        """,
        "Look for a stable reduced state model, a unitary scaffold, and the polar-factor error.  This is a diagnostic/initialization bridge, not a matrix AAK/Nehari solver.",
    ),
    Tutorial(
        "experimental_mimo_matrix_lattice_realization.py",
        "Experimental MIMO state-space to matrix-lattice realization",
        "Model-reduction tutorials",
        "Fit a stable matrix-lattice all-pass scaffold to the polar factor of a reduced MIMO state-space response.",
        """
        This tutorial promotes the previous bridge diagnostic into a small experimental solver-style
        API.  A stable MIMO state-space model is first reduced with the finite block-Hankel baseline.
        The experimental matrix-lattice helper then builds Markov-initialized all-pass scaffolds,
        searches over reflection gains, and returns the lattice with the smallest error against the
        reduced model's unitary polar factor.

        The result is a useful matrix-lattice realization scaffold and initialization diagnostic.
        Optional static gain compensation reports how much of the remaining mismatch can be explained
        by constant left/right gains around the all-pass lattice.  It is still not a full matrix
        AAK/Nehari solver and it does not realize arbitrary dynamic gain responses exactly.
        """,
        r"""
        For a square MIMO response ``H(e^{j\omega})``, the target is its polar factor

        .. math::

           U_p(e^{j\omega}) = U(e^{j\omega})V(e^{j\omega})^H,
           \qquad H=U\Sigma V^H.

        Candidate matrix-lattice all-pass responses ``G_\alpha`` are built from Markov directions
        and reflection gain ``\alpha``.  Static gain diagnostics then fit

        .. math::

           H(e^{j\omega}) \approx L\,G_\alpha(e^{j\omega})\,R.
        """,
        "Look for a stable lattice, unitarity error near numerical precision, selected gain, polar-factor fit error, and static-gain compensated error.  Treat this as an experimental all-pass realization scaffold.",
    ),
    Tutorial(
        "experimental_mimo_matrix_lattice_calibration.py",
        "Calibrating matrix-lattice static gain diagnostics",
        "Model-reduction tutorials",
        "Use a known matrix-lattice all-pass response to validate static gain compensation diagnostics.",
        """
        This tutorial is a calibration case for the experimental realization scaffold.  It starts
        from a known matrix-lattice all-pass response, wraps it in static nonunitary gains, and then
        fits static left/right gains back around the lattice response.  The compensated error should
        collapse when the only mismatch is static gain.
        """,
        r"""
        Given a lattice all-pass response ``G(e^{j\omega})`` and a target

        .. math::

           H(e^{j\omega}) = L\,G(e^{j\omega})\,R,

        the diagnostic solves a least-squares problem for static matrices ``L`` and ``R``.  This
        separates static nonunitary gain mismatch from dynamic lattice/all-pass mismatch.
        """,
        "Look for near-zero all-pass calibration error and a large improvement after fitting static gains around the gain-wrapped lattice response.",
    ),
    Tutorial(
        "matrix_lattice_allpass.py",
        "Matrix lattice all-pass response",
        "Multichannel and matrix tutorials",
        "Build a matrix all-pass lattice and verify unitary response behavior.",
        """
        Matrix lattice filters generalize scalar reflection coefficients to matrix reflection
        coefficients.  They are useful for compact multichannel unitary or paraunitary transforms.
        """,
        r"""
        A matrix-lattice section replaces a scalar reflection coefficient with a
        reflection matrix :math:`K_i`.  The contractive-stage diagnostic is

        .. math::

           \lVert K_i\rVert_2 < 1.

        Under the all-pass/scattering construction used here, the resulting transfer
        matrix :math:`G(z)` should satisfy, on the unit circle,

        .. math::

           G(e^{j\omega})^H G(e^{j\omega}) = I.

        Equivalently, every singular value of :math:`G(e^{j\omega})` should be one.
        The entry-magnitude plot shows how individual channel responses vary with
        frequency; the singular-value and residual plots check the stronger unitary
        matrix property.
        """,
        "The singular-value and unitarity-residual figures should be flat at one and near numerical precision if the generated matrix reflections are contractive.",
        runtime_status="""This example verifies the all-pass transfer function on a frequency grid and also uses ``MatrixLatticeAllPass.to_online_filter()`` to realize the same object as a sample-by-sample causal runtime.  The streaming impulse-response check uses only current input and previous section states.""",
    ),
    Tutorial(
        "coupled_mimo_lattice_filter.py",
        "Coupled MIMO matrix-lattice filtering",
        "Multichannel and matrix tutorials",
        "Apply a matrix-lattice all-pass to a coupled complex MIMO signal block and verify streaming energy preservation.",
        """
        This tutorial moves from static frequency-response diagnostics to a signal-processing use
        case.  A coupled complex multichannel signal is transformed by the causal
        ``OnlineMatrixLatticeAllPass`` runtime.  A finite-record time-domain adjoint then checks
        reconstruction.  The example verifies that the matrix-lattice response preserves energy
        while still mixing channels in a frequency-dependent way.
        """,
        r"""
        The matrix-lattice response :math:`G(z)` is designed as an all-pass multichannel
        transform:

        .. math::

           G(e^{j\omega})^H G(e^{j\omega}) = I.

        The forward online runtime applies the causal convolution

        .. math::

           y[n] = \sum_{k\ge 0} H_k x[n-k],

        where :math:`H_k\in\mathbb{C}^{c\times c}` are matrix impulse-response
        coefficients.  Energy preservation holds on the full stream, including the decaying
        all-pass tail:

        .. math::

           \sum_n \lVert y[n]\rVert_2^2
           \approx \sum_n \lVert x[n]\rVert_2^2.

        The finite-record synthesis diagnostic applies the time-domain adjoint

        .. math::

           x_{adj}[n] = \sum_{k\ge 0} H_k^H y[n+k].

        This adjoint is noncausal as a streaming inverse because it needs future transformed
        samples, but it is useful when the whole record is available.
        """,
        "Look for near-zero unitarity, energy, streaming-vs-impulse, and finite-adjoint reconstruction errors.  The covariance plots show that the streaming block is coupled even though it is norm preserving.",
        runtime_status="""The forward analysis path is causal and sample-by-sample.  The reconstruction check is a finite-record time-domain adjoint, so it is noncausal/transductive by design and should not be confused with a causal stable inverse.""",
    ),
    Tutorial(
        "matrix_unitary_response_compression.py",
        "Compressing a frequency-dependent unitary response",
        "Multichannel and matrix tutorials",
        "Represent a dense matrix-valued frequency response with a compact matrix lattice.",
        """
        Dense frequency grids can require many numbers.  A matrix lattice can represent a structured
        unitary response with far fewer parameters when the response is compatible with the model.
        """,
        r"""
        A dense sampled response stores one matrix per frequency bin,

        .. math::

           \{G_\ell\}_{\ell=0}^{L-1},
           \qquad G_\ell\in\mathbb{C}^{c\times c},

        which costs :math:`O(Lc^2)` scalar values.  A matrix-lattice model stores a
        small number of section matrices :math:`K_1,\ldots,K_m`, roughly
        :math:`O(mc^2)` scalar values for fixed realization details.

        The compression diagnostic is therefore

        .. math::

           \text{compression}
             = \frac{\text{dense scalar count}}{\text{lattice scalar count}}.

        The approximation is useful only if the compact representation still matches
        the target response and remains nearly unitary:

        .. math::

           \sigma_i(G(e^{j\omega})) \approx 1.
        """,
        "Compare the storage plot with the singular-value plot: the representation is compact, the frequency response should still look unitary, and the streaming trace confirms that the compact object is also a causal time-domain filter.",
        runtime_status="""This example still uses a dense grid to illustrate storage compression, but it also runs the same ``MatrixLatticeAllPass`` through ``to_online_filter()`` on a vector sequence.  The grid is a diagnostic; the compact lattice object has a causal streaming runtime.""",
    ),
    Tutorial(
        "paraunitary_filter_bank_demo.py",
        "Paraunitary filter-bank behavior",
        "Multichannel and matrix tutorials",
        "Demonstrate streaming analysis and finite-record adjoint reconstruction for a paraunitary-style transform.",
        """
        Paraunitary systems preserve energy and enable perfect reconstruction in multirate/filter-bank
        contexts.  Matrix lattice structures are a natural way to parameterize such systems.
        Here the forward analysis transform is run by the causal online lattice runtime, while
        synthesis is a finite-record time-domain adjoint diagnostic.
        """,
        r"""
        A paraunitary filter bank is most naturally described by its polyphase or
        frequency-response matrix :math:`E(z)`.  On the unit circle, the ideal
        condition is

        .. math::

           E(e^{j\omega})^H E(e^{j\omega}) = I.

        The online analysis path realizes the causal convolution

        .. math::

           y[n] = \sum_{k\ge 0} E_k x[n-k],

        and full-stream energy preservation is checked after appending a zero-input tail.
        The finite-record synthesis check applies the adjoint

        .. math::

           x_{adj}[n] = \sum_{k\ge 0} E_k^H y[n+k].

        The adjoint is time-domain but noncausal because it needs future analysis samples.
        This is the correct distinction between streaming analysis and block/transductive
        reconstruction.
        """,
        "The channel-energy, reconstruction-error, singular-value, and streaming-trace figures should show causal norm-preserving analysis plus near-perfect finite-adjoint reconstruction.",
        runtime_status="""The forward analysis transform is causal and streaming.  The synthesis/reconstruction check is a time-domain finite-block adjoint; it is noncausal because a stable causal all-pass generally has a noncausal stable inverse.""",
    ),
    Tutorial(
        "ml_unitary_convolution_demo.py",
        "Unitary convolution block for ML-style stability",
        "Multichannel and matrix tutorials",
        "Show a streaming norm-preserving convolution-like block motivated by stable ML layers.",
        """
        Orthogonal/unitary transforms can improve numerical stability in learned models.  This demo
        connects matrix-lattice ideas to norm-preserving convolution blocks as a DSP demonstration,
        not a full ML framework.  Unlike a circular FFT layer, the forward map here is run by the
        causal online matrix-lattice runtime.
        """,
        r"""
        The streaming block applies a causal multichannel convolution

        .. math::

           y[n] = \sum_{k\ge 0} H_k x[n-k].

        The all-pass condition

        .. math::

           H(e^{j\omega})^H H(e^{j\omega}) = I

        keeps the induced :math:`\ell_2` norm controlled on the full stream:

        .. math::

           \lVert y\rVert_2 \approx \lVert x\rVert_2,

        after appending enough zero-input samples to include the tail.  The finite-record
        adjoint diagnostic uses

        .. math::

           x_{adj}[n] = \sum_{k\ge 0} H_k^H y[n+k],

        which is useful for reconstruction checks but is noncausal as an online inverse.
        """,
        "Check the input/output norm figure, singular-value plot, streaming trace, and finite-adjoint error plot; a streaming unitary convolution block should preserve each batch-item norm after its tail is included.",
        runtime_status="""The forward map is causal and streaming.  The adjoint reconstruction check is time-domain but finite-block/noncausal, which matches how adjoints are used in offline ML-style diagnostics.""",
    ),
    Tutorial(
        "multichannel_audio_decorrelator.py",
        "Multichannel audio decorrelation with energy preservation",
        "Multichannel and matrix tutorials",
        "Reduce channel correlation while keeping total signal energy roughly unchanged.",
        """
        Decorrelators are useful for spatial audio and multichannel processing demos.  The goal here
        is not perceptual tuning; it is to show that matrix all-pass/lattice transforms can change
        correlation structure without changing total power much.
        """,
        r"""
        Let :math:`x[n]\in\mathbb{R}^c` be the input block and :math:`y[n]` the
        decorrelated output.  The sample covariance matrices are

        .. math::

           R_x = \mathbb{E}\{x[n]x[n]^T\},
           \qquad
           R_y = \mathbb{E}\{y[n]y[n]^T\}.

        Decorrelation aims to reduce off-diagonal covariance terms, for example

        .. math::

           r_{off}(R)=
           \frac{\sum_{i\ne j}|R_{ij}|}{\sum_i |R_{ii}|}.

        The transform is chosen to be approximately all-pass/unitary, so total energy
        should remain nearly unchanged:

        .. math::

           \frac{\sum_n\lVert y[n]\rVert_2^2}
                {\sum_n\lVert x[n]\rVert_2^2}
           \approx 1.

        The correlation heatmaps show the before/after coupling, while the energy
        ratio checks that decorrelation did not simply attenuate the signal.
        """,
        "Use the before/after correlation matrices and summary bar plot to see the decorrelation effect; the energy ratio should remain close to one.",
        runtime_status="""This decorrelator uses the causal ``OnlineMatrixLatticeAllPass`` runtime.  Each output frame depends only on the current input frame and previous lattice states.  The reported finite-block energy ratio can differ slightly from one because short prefixes omit the decaying all-pass tail.""",
    ),
    Tutorial(
        "tangential_schur_pick_jinner.py",
        "Tangential Schur Pick and J-inner diagnostics",
        "Tangential Schur and J-inner tutorials",
        "Check definite right-tangential Schur data with a Pick matrix and verify elementary Potapov J-inner factors.",
        """
        Tangential interpolation asks a matrix-valued Schur function to match only
        selected input directions rather than every column of the transfer matrix.
        This tutorial uses data generated from a known constant contraction so the
        Pick certificate, interpolation residual, and J-inner diagnostics can all be
        checked without hiding a difficult synthesis problem inside the example.
        """,
        r"""
        The right tangential data are

        .. math::

           S(z_i)u_i = v_i, \qquad \lVert S\rVert_\infty \le 1.

        The RKHS/de Branges--Rovnyak kernel for a Schur function is

        .. math::

           K_S(z,w)=\frac{I-S(z)^HS(w)}{1-\overline z w}.

        Testing this kernel on tangential directions gives the finite Pick Gram
        matrix

        .. math::

           P_{ij}=u_i^H K_S(z_i,z_j)u_j
                 =\frac{u_i^H u_j-v_i^H v_j}{1-\overline{z_i}z_j}.

        Feasibility in the definite problem requires :math:`P\succeq0`.
        For each strict rank-one datum, the graph vector
        :math:`\xi_i=[v_i;u_i]` has negative J-norm for

        .. math::

           J=\begin{bmatrix}I_{out}&0\\0&-I_{in}\end{bmatrix}.

        The scalar Blaschke factor and its Potapov--Blaschke lift are

        .. math::

           b_a(z)=\frac{z-a}{1-\overline a z},
           \qquad
           \Theta_i(z)=I+(b_{z_i}(z)-1)P_{\xi_i}.

        The projection :math:`P_{\xi_i}` is J-orthogonal.  The resulting
        :math:`\Theta_i` is J-inner on the unit circle and annihilates
        :math:`\xi_i` at :math:`z_i`.
        """,
        "The Pick eigenvalues should be nonnegative, the constant solution residual should be near roundoff, and the J-inner residual should be near numerical precision.",
        runtime_status="""This is finite interpolation-data analysis, not streaming filtering.  It is an offline Pick/J-inner diagnostic that explains the matrix-lattice background.""",
    ),
    Tutorial(
        "diagonal_tangential_schur_equals_scalar.py",
        "Diagonal tangential Schur equals independent scalar Pick problems",
        "Tangential Schur and J-inner tutorials",
        "Show that diagonal MIMO tangential data decompose into independent scalar Schur/Pick checks.",
        """
        This tutorial mirrors the diagonal-MIMO-equals-SISO runtime check in the
        interpolation setting.  If tangential directions are coordinate vectors and
        the Schur function is diagonal, the MIMO Pick matrix splits into independent
        scalar Pick blocks.  This gives a simple sanity check for the matrix-valued
        tangential machinery.
        """,
        r"""
        For a diagonal matrix Schur function

        .. math::

           S(z)=\operatorname{diag}(s_1(z),\ldots,s_c(z)),

        coordinate tangential data satisfy

        .. math::

           S(z_i)e_k = s_k(z_i)e_k.

        If the data are grouped by channel, the full Pick matrix is block diagonal,
        and each block is the scalar Pick matrix for one :math:`s_k`.
        """,
        "The full MIMO Pick matrix should match the scalar block-diagonal Pick matrix up to roundoff, and the diagonal constant solution should interpolate exactly.",
        runtime_status="""This is an offline interpolation diagnostic.  It does not filter a time stream; it checks that the MIMO Schur/Pick machinery reduces to independent scalar problems when there is no channel coupling.""",
    ),
    Tutorial(
        "echo_cancellation_erle_demo.py",
        "Synthetic ERLE metric demo",
        "Synthetic metric tutorials",
        "Explain ERLE and MSE on a controlled synthetic echo-path identification problem.",
        """
        This tutorial is intentionally limited: it is a metric demonstration for
        a synthetic echo path, not a production acoustic echo canceller.  It is
        useful for understanding the two-signal roles: a far-end/reference signal
        drives the unknown echo path, while a microphone/desired signal is used
        to measure the residual after subtracting the estimated echo.
        """,
        r"""
        With reference signal :math:`x[n]` and microphone signal :math:`m[n]`, an
        adaptive echo-path model produces

        .. math::

           \widehat e_{echo}[n] = h_{\theta_n}(x)[n],
           \qquad
           r[n] = m[n] - \widehat e_{echo}[n].

        Echo return loss enhancement is commonly reported as

        .. math::

           \operatorname{ERLE}_{dB}
             =10\log_{10}\frac{\mathbb{E}[m^2]}{\mathbb{E}[r^2]}.

        A causal echo-path filter may use current and past reference samples and
        previous adaptive state.  It must not use future microphone samples to
        form the current residual.
        """,
        "Higher ERLE and lower MSE indicate better cancellation in this controlled synthetic setup only; they are not product-quality acoustic echo-cancellation claims.",
        runtime_status="""This is a controlled two-signal synthetic metric demo.  It uses reference/far-end and microphone/desired roles, unlike one-signal AR prediction.  Real AEC systems require additional delay control, double-talk handling, nonlinear echo handling, and deployment-specific engineering.""",
    ),
    Tutorial(
        "channel_equalization_toy.py",
        "Toy channel equalization",
        "Applications and signal-model tutorials",
        "Small fixed-denominator equalization/system-identification prototype.",
        """
        This example is a compact equalization-style demo.  It keeps the setting small so readers can
        inspect the signal, filter, and error terms without needing a communication-system framework.
        """,
        r"""
        Equalization tries to learn an inverse or compensating filter so the output approaches a
        desired reference signal.
        """,
        "Inspect the printed MSE or improvement metric and remember this is a toy diagnostic, not a modem benchmark.",
    ),
    Tutorial(
        "system_identification.py",
        "Basic system identification sanity check",
        "Applications and signal-model tutorials",
        "Run a minimal identification example to check package wiring and API behavior.",
        """
        This page is intentionally simple.  It is a first script to run after installation to make
        sure the compiled extension and Python package import correctly.
        """,
        r"""
        The desired signal is generated by a known system from a reference input:

        .. math::

           d[n] = H(q^{-1})x[n].

        The model output :math:`\widehat d[n]` is compared with :math:`d[n]`.
        This is the two-signal identification form, not causal self-prediction of
        ``x[n]`` from its own past.
        """,
        "A successful run should complete without import/build errors and should report a finite identification error.",
        runtime_status="""The example is a small causal system-identification sanity check: output is formed from the reference input and current filter state, then compared with the desired signal.""",
    ),
)

BENCHMARKS: tuple[BenchmarkTutorial, ...] = (
    BenchmarkTutorial(
        "core_filtering",
        "run_benchmarks.py",
        "Core filtering and OpenMP batch benchmark",
        "Compare scalar/batch lattice filtering against SciPy lfilter baselines.",
        """
        This benchmark measures the core C++ filtering paths.  It is most meaningful for many
        independent streams because the batch path can amortize Python overhead and use OpenMP.
        """,
        r"""
        The speedup reported in the JSON is computed from median runtimes:

        .. math::

           \text{speedup} = \frac{t_{\text{baseline}}}{t_{\text{method}}}.
        """,
        "Single-stream SciPy may remain competitive.  The important comparison is batch C++/OpenMP versus batch SciPy for many channels.",
        (
            "--channels",
            "32",
            "--samples",
            "20000",
            "--repeats",
            "3",
            "--output",
            "core-filtering.json",
        ),
    ),
    BenchmarkTutorial(
        "model_reduction",
        "model_reduction_benchmark.py",
        "Model reduction speed/accuracy benchmark",
        "Reduce a full-order all-pole model and measure speed, error, SNR, and pole radius.",
        """
        Model reduction is a tradeoff: lower order is cheaper, but it may no longer match the
        original response.  This benchmark is intentionally a stable baseline, not a full
        Nehari/AAK/Hankel-norm reducer.  The lattice parameterization makes simple reflection
        truncation useful because it preserves scalar stability, while the theory documentation
        explains how Hankel-operator diagnostics connect to SISO reduction quality.
        """,
        r"""
        The benchmark reports relative MSE and SNR,

        .. math::

           \operatorname{relMSE}=\frac{\|y_{full}-y_{reduced}\|_2^2}{\|y_{full}\|_2^2},
           \qquad
           \operatorname{SNR}=10\log_{10}\frac{\mathbb{E}[y_{full}^2]}{\mathbb{E}[(y_{full}-y_{reduced})^2]}.
        """,
        "Look for the smallest order whose SNR and relative MSE are acceptable while keeping max pole radius below one.  Treat this as a stable baseline, not as a Hankel/Nehari/AAK optimality claim.",
        (
            "--full-order",
            "16",
            "--orders",
            "2",
            "4",
            "8",
            "12",
            "16",
            "--channels",
            "32",
            "--samples",
            "20000",
            "--repeats",
            "3",
            "--output",
            "model-reduction.json",
        ),
    ),
    BenchmarkTutorial(
        "hankel_reduction_speedup",
        "hankel_reduction_speedup.py",
        "Finite Hankel reduction amortization benchmark",
        "Measure when a one-time finite-Hankel reduction pays off during repeated high-order IIR filtering.",
        """
        The finite-Hankel reducer is a preprocessing step.  This benchmark makes the speed argument
        explicit by measuring reduction time, full-order filtering time, reduced-order filtering
        time, and the break-even number of samples per channel.  It separates the
        method is finite-Hankel/Ho--Kalman reduction, not an exact Nehari/AAK solver.
        """,
        r"""
        The break-even sample count is estimated as

        .. math::

           N_{break-even}
           = \frac{t_{reduce}}
                  {t_{full/sample}-t_{reduced/sample}}.
        """,
        "Look for high filter speedup, acceptable SNR/error, stable reduced denominators, and a break-even count that is small relative to the intended workload.",
        (
            "--full-orders",
            "16",
            "32",
            "--reduced-orders",
            "4",
            "8",
            "12",
            "--channels",
            "32",
            "--samples",
            "20000",
            "--repeats",
            "3",
            "--n-impulse",
            "512",
            "--hankel-rows",
            "64",
            "--hankel-cols",
            "64",
            "--output",
            "hankel-reduction-speedup.json",
        ),
    ),
    BenchmarkTutorial(
        "finite_aak_iir_reduction_speedup",
        "finite_aak_iir_reduction_speedup.py",
        "Finite-section AAK/Nehari IIR reduction benchmark",
        "Compare finite-Hankel and finite-section AAK/Nehari candidate reductions on the same stable SISO IIR filters.",
        """
        The finite-Hankel reducer and the finite-section AAK/Nehari candidate workflow are both
        useful baselines.  This benchmark runs them side by side on compressible stable SISO IIR
        filters and measures the practical tradeoff: reduction cost, filtering speedup,
        end-to-end speedup including reduction, SNR, magnitude-response error, pole radius, and
        break-even samples per channel.

        The benchmark is deliberately finite-section.  It is not a claim of exact
        infinite-dimensional AAK/Nehari optimality; it is a reproducible comparison of the mature
        baselines currently implemented in the package.
        """,
        r"""
        The end-to-end speedup includes the one-time reduction cost,

        .. math::

           S_{end-to-end} = \frac{t_{full}}{t_{reduce}+t_{reduced}}.

        The break-even sample count estimates when preprocessing has paid for itself,

        .. math::

           N_{break-even}
           = \frac{t_{reduce}}
                  {t_{full/sample}-t_{reduced/sample}}.
        """,
        "Look for stable reduced models with useful SNR/magnitude error and end-to-end speedup above one for the intended signal length.",
        (
            "--full-orders",
            "8",
            "16",
            "--target-orders",
            "3",
            "4",
            "6",
            "8",
            "--channels",
            "16",
            "--samples",
            "12000",
            "--repeats",
            "2",
            "--n-impulse",
            "384",
            "--hankel-rows",
            "48",
            "--hankel-cols",
            "48",
            "--output",
            "finite-aak-iir-reduction-speedup.json",
        ),
    ),
    BenchmarkTutorial(
        "mimo_hankel_reduction_speedup",
        "mimo_hankel_reduction_speedup.py",
        "Coupled MIMO finite-Hankel reduction benchmark",
        "Measure finite block-Hankel reduction cost and repeated state-space simulation speedup on coupled MIMO systems.",
        """
        The MIMO reducer returns state-space matrices rather than scalar filter coefficients.  This
        benchmark therefore measures the repeated cost of simulating the full and reduced MIMO
        systems on batched multichannel input signals.  It uses the compiled
        ``mimo_state_space_process_batch`` kernel when available, so the measured processing time
        reflects the current C++ state-space runtime rather than a pure Python loop.

        The table deliberately separates three concepts: processing speedup, one-shot end-to-end
        speedup including a single reduction, and amortized end-to-end speedup after reusing the
        reduced model for ``--reuse-count`` additional batches.  This keeps the benchmark scope explicit: the
        reduction can have excellent repeated-runtime speedups while still needing enough reuse to
        pay back preprocessing.

        This is still the reference block-Hankel/ERA-style baseline.  It is not a matrix AAK/Nehari
        solver; it is the finite block-Hankel reference point for comparison with matrix optimal-reduction methods.
        """,
        r"""
        The benchmark reports processing speedup

        .. math::

           S_{process}=\frac{t_{full}}{t_{reduced}},

        one-shot end-to-end speedup including one reduction,

        .. math::

           S_{one-shot}=\frac{t_{full}}{t_{reduce}+t_{reduced}},

        and amortized end-to-end speedup across ``K`` reused batches,

        .. math::

           S_{amortized}=\frac{K t_{full}}{t_{reduce}+K t_{reduced}}.
        """,
        "Look for stable reduced state matrices, decreasing Markov/output error with order, high processing speedup, and amortized end-to-end speedup above one when the workload reuses the reduced model enough times.",
        (
            "--full-orders",
            "8",
            "16",
            "--reduced-orders",
            "2",
            "4",
            "6",
            "8",
            "--inputs",
            "3",
            "--outputs",
            "3",
            "--batch",
            "8",
            "--samples",
            "6000",
            "--repeats",
            "2",
            "--reuse-count",
            "50",
            "--n-threads",
            "1",
            "--n-markov",
            "256",
            "--block-rows",
            "32",
            "--block-cols",
            "32",
            "--output",
            "mimo-hankel-reduction-speedup.json",
        ),
    ),
    BenchmarkTutorial(
        "matrix_lattice_runtime",
        "matrix_lattice_runtime.py",
        "Matrix-lattice all-pass runtime benchmark",
        "Compare compiled matrix-lattice frequency-response evaluation with the NumPy reference evaluator.",
        """
        Matrix lattice filters are most often used as compact frequency-dependent multichannel
        all-pass/scattering responses.  This benchmark measures the response-evaluation runtime
        for different channel dimensions and lattice orders, comparing the compiled C++ evaluator
        with the small NumPy reference implementation.
        """,
        r"""
        The benchmark reports

        .. math::

           S = \frac{t_{NumPy}}{t_{compiled}},

        along with the relative difference between implementations and the maximum unitarity error
        over the frequency grid.
        """,
        "Look for relative differences near numerical precision, small unitarity error, and speedups above one for larger frequency grids/orders.",
        (
            "--dims",
            "2",
            "3",
            "4",
            "--orders",
            "2",
            "4",
            "8",
            "--n-freq",
            "1024",
            "--repeats",
            "2",
            "--n-threads",
            "1",
            "--output",
            "matrix-lattice-runtime.json",
        ),
    ),
    BenchmarkTutorial(
        "tangential_schur_mimo",
        "tangential_schur_mimo_benchmark.py",
        "MIMO tangential-Schur/Pick and J-inner benchmark",
        "Measure finite MIMO tangential Pick, constant Schur recovery, and Potapov/J-inner diagnostic costs.",
        """
        The tangential-Schur layer is a mathematical bridge between MIMO
        interpolation, J-inner/lossless systems, and the matrix-lattice examples.
        This benchmark measures the finite definite baseline: constructing the
        right-tangential Pick matrix, checking its Hermitian spectrum, recovering
        a compatible constant Schur matrix, and evaluating elementary Potapov
        J-inner products on a boundary grid.

        The benchmark also includes a diagonal-MIMO sanity case.  When tangential
        directions are coordinate vectors and the Schur map is diagonal, the full
        MIMO Pick matrix decomposes into independent scalar Pick blocks.  The
        benchmark reports both numerical block/eigenvalue agreement and the cost
        difference between treating the problem as dense MIMO data versus scalar
        subproblems.
        """,
        r"""
        For right tangential data

        .. math::

           S(z_i) U_i = V_i,

        the Pick blocks are

        .. math::

           P_{ij} = \frac{U_i^H U_j - V_i^H V_j}{1 - \overline{z_i} z_j}.

        A finite definite Schur-class certificate requires :math:`P \succeq 0`.
        The J-inner diagnostic checks elementary Potapov products on the unit
        circle:

        .. math::

           \Theta(e^{j\omega})^H J \Theta(e^{j\omega}) \approx J.

        In the diagonal case, coordinate directions should make the dense MIMO
        Pick matrix equal a block diagonal assembly of scalar Pick matrices.
        """,
        "Use the runtime-breakdown figure to see where cost goes, the residual figure to check interpolation and J-inner accuracy, and the diagonal comparison to verify reduction to independent scalar Schur problems.",
        (
            "--dims",
            "2",
            "3",
            "4",
            "--points",
            "4",
            "8",
            "--diagonal-points-per-channel",
            "4",
            "8",
            "--boundary-grid",
            "256",
            "--repeats",
            "2",
            "--output",
            "tangential-schur-mimo.json",
            "--csv-output",
            "tangential-schur-mimo.csv",
        ),
    ),
    BenchmarkTutorial(
        "experimental_mimo_matrix_lattice_realization_sweep",
        "experimental_mimo_matrix_lattice_realization_sweep.py",
        "Experimental MIMO matrix-lattice realization sweep",
        "Sweep reduced and lattice orders for the experimental all-pass/polar matrix-lattice scaffold.",
        """
        The experimental state-space to matrix-lattice helper returns an all-pass scaffold, not a
        full dynamic gain realization.  This benchmark runs that helper over a small grid of
        reduced MIMO orders and lattice orders, then reports polar-factor error, raw state-response
        error, static-gain-compensated error, gain conditioning, unitarity, and a diagnostic
        classification.

        This makes the current matrix-lattice realization story measurable: good all-pass fits,
        mostly static-gain mismatch, and poor lattice-scaffold fits are separated explicitly.
        """,
        r"""
        Static gain compensation fits

        .. math::

           H(e^{j\omega}) \approx L\,G_{lattice}(e^{j\omega})\,R.

        The improvement ratio is

        .. math::

           I = \frac{\|H-G\|_F}{\|H-LGR\|_F}.
        """,
        "Look for unitary lattice responses, stable reduced states, lower static-gain-compensated error than raw state-response error, and classifications that explain whether the mismatch is all-pass, static-gain, or scaffold-driven.",
        (
            "--full-order",
            "12",
            "--reduced-orders",
            "4",
            "6",
            "--lattice-orders",
            "2",
            "4",
            "--channels",
            "3",
            "--n-markov",
            "128",
            "--n-freq",
            "128",
            "--block-rows",
            "20",
            "--block-cols",
            "20",
            "--candidate-gains",
            "0.20",
            "0.40",
            "0.60",
            "0.80",
            "--static-gain-iterations",
            "10",
            "--repeats",
            "1",
            "--n-threads",
            "1",
            "--output",
            "experimental-mimo-matrix-lattice-realization-sweep.json",
        ),
    ),
    BenchmarkTutorial(
        "finite_nehari_rank_sweep",
        "finite_nehari_rank_sweep.py",
        "Finite Nehari/AAK rank-sweep benchmark",
        "Measure how finite Hankel singular values and structured tail errors change with approximation rank.",
        """
        This benchmark is the conservative bridge between the finite-Hankel reducer and deeper
        Nehari/AAK theory.  It builds a finite Hankel matrix from an anticausal tail, runs
        ``finite_nehari_approximate_tail`` for several ranks, and reports both the unconstrained
        SVD error and the Hankelized structured approximation error.

        The benchmark does not claim exact infinite-dimensional Nehari or AAK optimality.  It is a
        numerical validation benchmark that makes the rank/error tradeoff visible within
        the documented finite-section scope.
        """,
        r"""
        For the finite matrix problem, the Eckart--Young theorem gives

        .. math::

           \min_{\operatorname{rank}(X)\le r} \|H-X\|_2 = \sigma_{r+1}(H).

        After anti-diagonal averaging, the approximation is Hankel-structured again but need not
        remain rank ``r``.  Its error is therefore reported as a separate diagnostic.
        """,
        "Look for monotone decay of sigma_{r+1}, agreement between unconstrained SVD error and sigma_{r+1}, and decreasing tail-relative error as rank increases.",
        (
            "--rows",
            "32",
            "--cols",
            "32",
            "--ranks",
            "1",
            "2",
            "3",
            "4",
            "6",
            "--output",
            "finite-nehari-rank-sweep.json",
            "--csv-output",
            "finite-nehari-rank-sweep.csv",
        ),
    ),
    BenchmarkTutorial(
        "block_levinson",
        "block_levinson_benchmark.py",
        "Block Levinson versus dense block-Toeplitz solve",
        "Compare structured vector AR estimation against a dense direct solve.",
        """
        This benchmark validates the multichannel AR implementation by comparing block
        Levinson-Durbin coefficients with a dense block-Toeplitz solve.  It is currently more of a
        numerical validation benchmark than a speedup claim.
        """,
        r"""
        The key numerical diagnostic is the coefficient difference norm between the two solutions.
        """,
        "Small coefficient differences validate the recursion.  Reflection spectral norms below one indicate a stable matrix-reflection sequence.",
        (
            "--channels",
            "4",
            "--order",
            "8",
            "--samples",
            "20000",
            "--repeats",
            "3",
            "--output",
            "block-levinson.json",
        ),
    ),
    BenchmarkTutorial(
        "adaptive_period_sweep",
        "adaptive_period_sweep.py",
        "Adaptive reflection update-period sweep",
        "Measure speed/quality effects of updating reflection coefficients less often.",
        """
        Adaptive IIR models can save work by updating denominator/reflection coefficients less
        frequently than numerator taps.  The benchmark sweeps the update period and writes both JSON
        and CSV outputs.
        """,
        r"""
        Larger update periods reduce update count.  The question is whether tail MSE remains close
        to the period-1 baseline.
        """,
        "Look for the largest period with a good speedup and modest tail-MSE degradation.",
        (
            "--periods",
            "1",
            "2",
            "4",
            "8",
            "16",
            "--samples",
            "12000",
            "--repeats",
            "3",
            "--output",
            "adaptive-period-sweep.json",
            "--csv-output",
            "adaptive-period-sweep.csv",
        ),
    ),
    BenchmarkTutorial(
        "echo_metric",
        "echo_cancellation_benchmark.py",
        "Synthetic echo metric benchmark",
        "Compare synthetic echo-path metrics across simple baselines and lattice-based variants.",
        """
        This benchmark is included to exercise metrics such as ERLE and residual MSE on a controlled
        synthetic problem.  It is not an acoustic echo cancellation product benchmark.
        """,
        r"""
        ERLE is

        .. math::

           10\log_{10}\frac{\mathbb{E}[d^2]}{\mathbb{E}[e^2]}.
        """,
        "Use ERLE and MSE only within this controlled synthetic setup; do not compare the numbers to production AEC systems.",
        (
            "--samples",
            "16000",
            "--sample-rate",
            "16000",
            "--repeats",
            "1",
            "--output",
            "echo-metric.json",
        ),
    ),
)


def clean(text: str) -> str:
    return textwrap.dedent(text).strip()


def slug(script: str) -> str:
    return Path(script).stem


def _repo_relative(path: Path, base: Path) -> str:
    """Return a portable path for generated docs and command output."""

    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _is_generated_artifact_log(line: str) -> bool:
    """Return True for non-essential "wrote ..." artifact messages."""

    stripped = line.strip()
    if not stripped.lower().startswith("wrote "):
        return False
    path_text = stripped.split(maxsplit=1)[1]
    return path_text.startswith(
        (
            "docs/examples/generated/_artifacts/",
            "docs/benchmarks/generated/_artifacts/",
        )
    )


def _sanitize_captured_output(text: str, repo_root: Path) -> str:
    """Hide machine-specific paths and noisy artifact logs from docs."""

    if not text:
        return ""
    root = repo_root.resolve()
    replacements = {
        root.as_posix() + "/": "",
        str(root) + os.sep: "",
        root.as_posix(): ".",
        str(root): ".",
    }
    cleaned = text
    for needle, repl in replacements.items():
        cleaned = cleaned.replace(needle, repl)
    lines = [line for line in cleaned.splitlines() if not _is_generated_artifact_log(line)]
    return "\n".join(lines).strip()


def run_command(
    command: list[str],
    *,
    cwd: Path,
    artifact_dir: Path,
    timeout: float,
) -> tuple[int | None, str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd) + os.pathsep + env.get("PYTHONPATH", "")
    # Use a repo-relative artifact directory so captured tutorial output does
    # not expose a contributor's local filesystem path.
    env["LATTICE_DSP_ARTIFACT_DIR"] = _repo_relative(artifact_dir, cwd)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return (
            proc.returncode,
            _sanitize_captured_output(proc.stdout, cwd),
            _sanitize_captured_output(proc.stderr, cwd),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        stderr = stderr + f"\nTimed out after {timeout} seconds."
        return None, _sanitize_captured_output(stdout, cwd), _sanitize_captured_output(stderr, cwd)


def write_page(
    path: Path,
    *,
    title: str,
    purpose: str,
    context: str,
    equations: str,
    readout: str,
    command: list[str],
    stdout: str,
    stderr: str,
    returncode: int | None,
    artifact_dir: Path,
    source_include: str,
    sample_output: str = "",
    runtime_status: str = "",
    verification: str = "",
    show_captured_output: bool = True,
    artifact_intro: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rel_artifact = artifact_dir.relative_to(path.parent)
    images = (
        sorted(p for p in artifact_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if artifact_dir.exists()
        else []
    )
    data_files = (
        sorted(p for p in artifact_dir.iterdir() if p.suffix.lower() in DATA_SUFFIXES)
        if artifact_dir.exists()
        else []
    )

    lines: list[str] = [title, "=" * len(title), ""]
    lines += [".. admonition:: Tutorial goal", "", f"   {purpose}", ""]
    lines += [
        ".. note::",
        "",
        "   New to the terminology? See the :doc:`lattice DSP concept map <../../algorithms/concept_map>` and the :doc:`causality/data-use guide <../../theory/causality_and_data_use>` for how online, offline, block, and MIMO examples should be read.",
        "",
    ]
    lines += ["Context", "-------", "", clean(context), ""]
    lines += ["Key idea and equations", "----------------------", "", clean(equations), ""]
    if runtime_status:
        lines += ["Causality and data use", "----------------------", "", clean(runtime_status), ""]
    if verification:
        lines += [
            "What this example verifies",
            "--------------------------",
            "",
            clean(verification),
            "",
        ]
    lines += ["How to read the result", "----------------------", "", clean(readout), ""]
    lines += ["Run command", "-----------", "", ".. code-block:: bash", ""]
    lines += ["   " + " ".join(command), ""]

    if sample_output:
        lines += [
            "Representative local output",
            "---------------------------",
            "",
            clean(sample_output),
            "",
        ]

    if returncode is not None:
        lines += ["Run status", "----------", "", f"Return code: ``{returncode}``", ""]

    if artifact_intro:
        lines += [
            "Visual and data readout",
            "-----------------------",
            "",
            clean(artifact_intro),
            "",
        ]

    if stdout and show_captured_output:
        lines += ["Captured stdout", "---------------", "", ".. code-block:: text", ""]
        lines += ["   " + line for line in stdout.splitlines()]
        lines.append("")
    if stderr and show_captured_output:
        lines += ["Captured stderr", "---------------", "", ".. code-block:: text", ""]
        lines += ["   " + line for line in stderr.splitlines()]
        lines.append("")

    if images:
        lines += ["Figures", "-------", ""]
        for img in images:
            lines += [
                f".. figure:: {rel_artifact.as_posix()}/{img.name}",
                "   :alt: " + img.stem.replace("_", " "),
                "   :width: 95%",
                "",
            ]
            lines += [f"   ``{img.name}``", ""]

    if data_files:
        lines += ["Generated data files", "--------------------", ""]
        for data in data_files:
            lines += [f"* :download:`{data.name} <{rel_artifact.as_posix()}/{data.name}>`"]
        lines.append("")

    lines += [
        "Source code",
        "-----------",
        "",
        f".. literalinclude:: {source_include}",
        "   :language: python",
        "   :linenos:",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


MIMO_VERIFICATION_NOTES = {
    "tangential_schur_pick_jinner.py": r"""
    This verifies the finite definite tangential-Schur baseline.  The Pick
    matrix should be positive semidefinite, the known constant contraction
    should interpolate the data to roundoff, elementary Potapov factors should
    annihilate their graph vectors, and the factor product should be J-inner on
    the unit circle.
    """,
    "diagonal_tangential_schur_equals_scalar.py": r"""
    This verifies the reduction-to-scalar sanity check for interpolation data.
    When tangential directions are coordinate vectors and the Schur function is
    diagonal, the full MIMO Pick matrix should split into independent scalar
    Pick blocks with roundoff-level off-block error.
    """,
    "mimo_model_reduction_stress_cases.py": r"""
    This verifies the finite MIMO model-reduction baseline on three deliberately
    different matrix impulse-response families: a slow 1/f-type tail, a large
    random rational 10-by-10 response, and an ill-conditioned high-degree 2-by-2
    response.  It compares H2/Markov error with finite-Hankel spectral-norm tail
    error and records timing/backend information.  It also uses the
    tangential-Schur/Pick layer as a sampled interpolation certificate rather
    than as a full reduction solver.
    """,
    "mimo_diagonal_equals_independent_siso.py": r"""
    This is the reduction-to-SISO sanity check.  It verifies that when the MIMO
    Markov matrices or matrix reflections are diagonal, the multichannel object
    decomposes into independent scalar systems.  The expected diagnostic is
    roundoff-level agreement between the MIMO result and stacked SISO results,
    including the online ``predict()``/``update()`` predictor path.
    """,
    "causal_mimo_lattice_prediction.py": r"""
    This verifies the online contract for ``MIMOLatticePredictor``.  Prediction
    is requested before the current vector is observed, then ``update(y_n)``
    consumes the current vector and advances the backward-error state.  The
    residual is compared with the direct VAR residual from the same fitted
    block-Levinson model.
    """,
    "online_coupled_mimo_vs_siso.py": r"""
    This verifies the reason to use MIMO rather than three independent SISO
    predictors.  On a held-out coupled VAR stream, the full matrix predictor
    should reduce residual RMS and, more importantly, reduce residual
    cross-channel correlation relative to diagonal/SISO baselines.
    """,
    "multichannel_levinson_ar.py": r"""
    This verifies the batch multichannel AR estimator.  The block Levinson result
    is compared with a dense block-Toeplitz solve, reflection spectral norms are
    checked, and the residual covariance shows what cross-channel prediction
    error remains after fitting.
    """,
    "matrix_ar_spectral_estimation.py": r"""
    This verifies the spectral interpretation of a fitted matrix AR model.  The
    plotted spectral-density matrix separates auto-spectra from cross-spectra,
    while companion radius and reflection norms diagnose whether the fitted model
    can be used as a stable causal VAR/IIR recursion.
    """,
    "mimo_lattice_vs_block_levinson.py": r"""
    This is a side-by-side diagnostic, not an equivalence proof.  It verifies the
    VAR side with block-Levinson prediction diagnostics and the matrix-lattice
    side with contractive reflection norms and frequency-wise unitarity.
    """,
    "matrix_lattice_allpass.py": r"""
    This verifies that the causal online all-pass runtime matches the frequency
    response represented by ``MatrixLatticeAllPass``.  The impulse-response FFT
    should agree with the frequency-grid evaluator, and singular values should
    remain close to one on the unit circle.
    """,
    "coupled_mimo_lattice_filter.py": r"""
    This verifies streaming coupled forward filtering.  The output is produced by
    the online matrix-lattice runtime, off-diagonal impulse/Markov energy shows
    channel coupling, and the finite-record adjoint is labeled separately as a
    noncausal reconstruction diagnostic.
    """,
    "multichannel_audio_decorrelator.py": r"""
    This verifies a causal forward decorrelator.  The online all-pass runtime
    should reduce off-diagonal covariance/correlation while preserving total
    energy after the filter tail is included.  It is a DSP diagnostic, not a
    perceptual audio product claim.
    """,
    "matrix_unitary_response_compression.py": r"""
    This verifies compact representation rather than online prediction.  The
    dense frequency response is compared with a smaller matrix-lattice
    parameterization; a streaming trace confirms that the same compact object can
    also run as a causal forward filter.
    """,
    "paraunitary_filter_bank_demo.py": r"""
    This verifies the split between streaming analysis and finite-record
    synthesis.  The forward paraunitary-style analysis is causal and
    norm-preserving after the tail is included; the adjoint reconstruction check
    is time-domain but noncausal/transductive.
    """,
    "ml_unitary_convolution_demo.py": r"""
    This verifies a DSP analogue of a norm-preserving convolution block.  The
    forward map is causal and streaming; norm preservation is checked on the
    full stream with tail padding, while the adjoint reconstruction diagnostic is
    finite-record and noncausal.
    """,
}


MIMO_TUTORIAL_ORDER = {
    "mimo_diagonal_equals_independent_siso.py": 0,
    "causal_mimo_lattice_prediction.py": 1,
    "online_coupled_mimo_vs_siso.py": 2,
    "multichannel_levinson_ar.py": 3,
    "matrix_ar_spectral_estimation.py": 4,
    "mimo_lattice_vs_block_levinson.py": 5,
    "matrix_lattice_allpass.py": 6,
    "coupled_mimo_lattice_filter.py": 7,
    "multichannel_audio_decorrelator.py": 8,
    "matrix_unitary_response_compression.py": 9,
    "paraunitary_filter_bank_demo.py": 10,
    "ml_unitary_convolution_demo.py": 11,
}


def _ordered_group_items(group: str, items: list[Tutorial]) -> list[Tutorial]:
    if group == "Multichannel and matrix tutorials":
        return sorted(items, key=lambda item: MIMO_TUTORIAL_ORDER.get(item.script, 1000))
    return items


def generate_examples(repo_root: Path, docs_dir: Path, *, run: bool, timeout: float) -> None:
    out_dir = docs_dir / "examples" / "generated"
    artifact_root = out_dir / "_artifacts"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[Tutorial]] = {}
    for item in EXAMPLES:
        if (repo_root / "examples" / item.script).exists():
            groups.setdefault(item.group, []).append(item)

    index_lines = [
        "Examples tutorials",
        "==================",
        "",
        "These pages are generated from the runnable scripts in ``examples/``.  Each tutorial page explains the context, shows the main equations, states what the example verifies when applicable, embeds generated figures/data, and includes the source code.  For terminology across the package, start with :doc:`../algorithms/concept_map`; for MIMO claims, see :doc:`../algorithms/mimo_verification_map`.",
        "",
        "Build the full tutorial gallery with results:",
        "",
        ".. code-block:: bash",
        "",
        "   ./scripts/build_docs_with_results.sh",
        "",
    ]
    for group, items in groups.items():
        ordered_items = _ordered_group_items(group, items)
        index_lines += [group, "-" * len(group), "", ".. toctree::", "   :maxdepth: 1", ""]
        for item in ordered_items:
            index_lines.append(f"   generated/{slug(item.script)}")
        index_lines.append("")
    (docs_dir / "examples" / "index.rst").write_text("\n".join(index_lines), encoding="utf-8")

    for item in EXAMPLES:
        script_path = repo_root / "examples" / item.script
        if not script_path.exists():
            continue
        page = out_dir / f"{slug(item.script)}.rst"
        art = artifact_root / slug(item.script)
        art.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, f"examples/{item.script}"]
        stdout = stderr = ""
        returncode: int | None = None
        if run:
            returncode, stdout, stderr = run_command(
                command, cwd=repo_root, artifact_dir=art, timeout=timeout
            )
        write_page(
            page,
            title=item.title,
            purpose=item.purpose,
            context=item.context,
            equations=item.equations,
            readout=item.readout,
            command=["python", f"examples/{item.script}"],
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            artifact_dir=art,
            source_include=f"../../../examples/{item.script}",
            sample_output=item.sample_output,
            runtime_status=item.runtime_status,
            verification=MIMO_VERIFICATION_NOTES.get(item.script, ""),
        )


def generate_benchmarks(repo_root: Path, docs_dir: Path, *, run: bool, timeout: float) -> None:
    out_dir = docs_dir / "benchmarks" / "generated"
    artifact_root = out_dir / "_artifacts"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)

    index_lines = [
        "Benchmark tutorials",
        "===================",
        "",
        "These pages are generated from scripts in ``benchmarks/``.  They explain what each benchmark measures, prefer visual summaries for quick reading, and keep JSON/CSV outputs under the generated artifact directory instead of the repository root.  For terminology across lattice filters, model reduction, and MIMO systems, see :doc:`../algorithms/concept_map`.",
        "",
        "Build the full benchmark tutorial gallery with results:",
        "",
        ".. code-block:: bash",
        "",
        "   ./scripts/build_docs_with_results.sh",
        "",
        ".. toctree::",
        "   :maxdepth: 1",
        "",
    ]
    for item in BENCHMARKS:
        if (repo_root / "benchmarks" / item.script).exists():
            index_lines.append(f"   generated/{item.slug}")
    index_lines.append("")
    (docs_dir / "benchmarks" / "index.rst").write_text("\n".join(index_lines), encoding="utf-8")

    for item in BENCHMARKS:
        script_path = repo_root / "benchmarks" / item.script
        if not script_path.exists():
            continue
        page = out_dir / f"{item.slug}.rst"
        art = artifact_root / item.slug
        art.mkdir(parents=True, exist_ok=True)

        # Benchmark scripts receive repo-relative output filenames.  This keeps
        # captured stdout portable in the generated public documentation.
        args = list(item.args)
        for flag in ["--output", "--csv-output", "--markdown-output", "--plot-output"]:
            if flag in args:
                idx = args.index(flag) + 1
                args[idx] = _repo_relative(art / args[idx], repo_root)
        command = [sys.executable, f"benchmarks/{item.script}", *args]
        stdout = stderr = ""
        returncode: int | None = None
        if run:
            returncode, stdout, stderr = run_command(
                command, cwd=repo_root, artifact_dir=art, timeout=timeout
            )
            create_benchmark_visuals(art, slug=item.slug, title=item.title)
        display_args = list(item.args)
        for flag in ["--output", "--csv-output", "--markdown-output", "--plot-output"]:
            if flag in display_args:
                idx = display_args.index(flag) + 1
                display_args[idx] = (
                    f"docs/benchmarks/generated/_artifacts/{item.slug}/{display_args[idx]}"
                )
        write_page(
            page,
            title=item.title,
            purpose=item.purpose,
            context=item.context,
            equations=item.equations,
            readout=item.readout,
            command=["python", f"benchmarks/{item.script}", *display_args],
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            artifact_dir=art,
            source_include=f"../../../benchmarks/{item.script}",
            show_captured_output=False,
            artifact_intro=(
                "When the benchmark gallery is built with results, this page embeds PNG "
                "summaries generated from the same JSON/CSV artifacts.  The raw data stay "
                "available below as downloads so exact numbers remain reproducible without "
                "making the public page read like console output."
            ),
        )


def generate_gallery(
    *,
    repo_root: Path,
    docs_dir: Path,
    run_examples: bool = False,
    run_benchmarks: bool = False,
    timeout: float = 120.0,
) -> None:
    generate_examples(repo_root, docs_dir, run=run_examples, timeout=timeout)
    generate_benchmarks(repo_root, docs_dir, run=run_benchmarks, timeout=timeout)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate tutorial-style Sphinx example/benchmark pages."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--run-examples", action="store_true")
    parser.add_argument("--run-benchmarks", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    generate_gallery(
        repo_root=args.repo_root.resolve(),
        docs_dir=args.docs_dir.resolve(),
        run_examples=args.run_examples,
        run_benchmarks=args.run_benchmarks,
        timeout=args.timeout,
    )
    print(f"Generated tutorial pages in {args.docs_dir}")


if __name__ == "__main__":
    main()
