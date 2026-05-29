"""Release-trust tests for the public 0.1 scope.

These tests exercise the claims that appear in the README and theory docs:
reflection-bounded scalar IIR stability, explicit adaptive-IIR failure modes,
finite SISO/MIMO model-reduction diagnostics, and small-scale smoke runs of the
flagship long-signal tutorials.  They intentionally avoid hard timing
thresholds; throughput numbers are machine-dependent.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import lattice_dsp as ld
from examples.stability_vs_direct_iir import direct_form_identifier, max_pole_radius


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_example(script: str, args: list[str], tmp_path: Path) -> str:
    env = os.environ.copy()
    env["LATTICE_DSP_ARTIFACT_DIR"] = str(tmp_path / "artifacts")
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / script), *args],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert proc.stdout.strip()
    return proc.stdout


def test_bounded_reflection_coefficients_remain_stable_near_boundary() -> None:
    """Scalar reflection coordinates should remain stable even near |k| = 1."""

    reflection = np.array([0.999, -0.995, 0.98, -0.94], dtype=np.float64)
    denominator = np.asarray(ld.reflection_to_denominator(reflection.tolist()), dtype=np.float64)
    poles = np.roots(denominator)

    assert np.all(np.isfinite(denominator))
    assert poles.size == reflection.size
    assert float(np.max(np.abs(poles))) < 1.0

    restored = np.asarray(ld.denominator_to_reflection(denominator.tolist()), dtype=np.float64)
    np.testing.assert_allclose(restored, reflection, atol=2e-10, rtol=2e-10)


def test_direct_iir_denominator_can_cross_unit_circle_while_lattice_stays_stable() -> None:
    """The public stability tutorial should exhibit the intended failure mode."""

    rng = np.random.default_rng(11)
    x = rng.normal(size=10_000)
    target_reflection = [0.82, -0.55]
    target_numerator = [0.35, -0.05, 0.6]
    desired = np.asarray(
        ld.LatticeIIR(target_reflection, target_numerator).process(x), dtype=np.float64
    )

    _, direct_error, direct_radii = direct_form_identifier(x, desired)
    assert np.nanmax(direct_radii) > 1.0
    assert np.isfinite(np.nanmean(direct_error[:1000] ** 2))

    lattice = ld.AdaptiveLatticeLadderNLMS(
        initial_reflection=[0.0, 0.0],
        initial_taps=[0.0, 0.0, 0.0],
        mu_taps=0.06,
        mu_reflection=0.0015,
        margin=1e-4,
    )
    lattice_error = np.asarray(lattice.adapt_block(x, desired), dtype=np.float64)
    lattice_radius = max_pole_radius(np.asarray(lattice.denominator, dtype=np.float64))

    assert np.all(np.abs(np.asarray(lattice.reflection, dtype=np.float64)) < 1.0 - 1e-4 + 1e-12)
    assert lattice_radius < 1.0
    assert np.isfinite(np.mean(lattice_error[-1000:] ** 2))


def test_siso_finite_hankel_reduction_returns_stable_finite_diagnostics() -> None:
    """SISO finite-Hankel reduction should return stable, finite diagnostics."""

    reflection = [0.4, -0.2, 0.1]
    numerator = [1.0, 0.3, -0.1, 0.05]
    result = ld.finite_hankel_reduce_iir(
        reflection,
        numerator,
        reduced_order=2,
        n_impulse=120,
        rows=24,
        cols=24,
    )

    assert result["method"] == "finite_hankel_ho_kalman"
    assert result["stable"] is True
    assert result["relative_impulse_error"] >= 0.0
    assert result["relative_impulse_error"] < 0.5
    assert np.all(np.isfinite(result["hankel_singular_values"]))
    assert len(result["denominator"]) == 3
    assert len(result["numerator"]) == 3


def test_mimo_block_hankel_reduction_returns_stable_shapes_and_error() -> None:
    """The rare MIMO path should preserve shape, stability, and diagnostics."""

    rng = np.random.default_rng(2029)
    order = 6
    inputs = 2
    outputs = 3
    q, _ = np.linalg.qr(rng.normal(size=(order, order)))
    A = q @ np.diag(np.linspace(0.82, 0.18, order)) @ q.T
    B = rng.normal(size=(order, inputs)) / np.sqrt(order)
    C = rng.normal(size=(outputs, order)) / np.sqrt(order)
    D = 0.02 * rng.normal(size=(outputs, inputs))

    markov = np.asarray(ld.mimo_state_space_markov_response(A, B, C, D, 80), dtype=np.float64)
    result = ld.finite_hankel_reduce_mimo(markov, reduced_order=4, block_rows=10, block_cols=10)
    approx = np.asarray(
        ld.mimo_state_space_markov_response(result["A"], result["B"], result["C"], result["D"], 80),
        dtype=np.float64,
    )

    assert result["A"].shape == (4, 4)
    assert result["B"].shape == (4, inputs)
    assert result["C"].shape == (outputs, 4)
    assert result["D"].shape == (outputs, inputs)
    assert result["stable"] is True
    assert result["retained_hankel_energy"] > 0.95
    assert np.all(np.isfinite(result["hankel_singular_values"]))
    relative_error = float(np.sum((markov - approx) ** 2) / max(np.sum(markov * markov), 1e-30))
    assert relative_error < 0.1


def test_causal_mimo_lattice_predictor_matches_fitted_var_residual() -> None:
    """The online MIMO lattice path should be causal after batch fitting."""

    rng = np.random.default_rng(2031)
    coefficients = np.asarray(
        [
            [[0.30, 0.06], [-0.04, 0.25]],
            [[-0.09, 0.02], [0.01, -0.07]],
        ],
        dtype=np.float64,
    )
    x = np.zeros((2500, 2), dtype=np.float64)
    noise = rng.normal(scale=0.3, size=x.shape)
    for n in range(2, x.shape[0]):
        x[n] = noise[n] - coefficients[0] @ x[n - 1] - coefficients[1] @ x[n - 2]

    r = ld.multichannel_autocorrelation(x, order=2)
    result = ld.block_levinson_durbin(r, order=2)
    predictor = ld.MIMOLatticePredictor.from_levinson(result)
    prediction, error = predictor.process(x)
    direct_error = ld.multichannel_prediction_error(x, result.coefficients)

    assert prediction.shape == x.shape
    assert error.shape == x.shape
    assert (
        np.linalg.norm(error[2:] - direct_error) / max(np.linalg.norm(direct_error), 1e-30) < 1e-12
    )
    assert np.allclose(error, x - prediction, atol=1e-12)
    assert np.max(result.reflection_spectral_norms) < 1.0


def test_million_sample_iir_tutorial_smoke_small_scale(tmp_path: Path) -> None:
    stdout = _run_example(
        "examples/million_sample_iir_throughput.py",
        ["--samples", "2048", "--tail-taps", "512", "--repeats", "1"],
        tmp_path,
    )
    assert "million-sample IIR throughput demonstration" in stdout
    assert "IIR throughput" in stdout
    assert "relative RMS error" in stdout


def test_large_echo_stress_tutorial_smoke_small_scale(tmp_path: Path) -> None:
    stdout = _run_example(
        "examples/large_order_echo_stress.py",
        ["--samples", "1024", "--order", "8", "--echo-taps", "256", "--repeats", "1"],
        tmp_path,
    )
    assert "large echo-scale stable recursive model stress" in stdout
    assert "stage update rate" in stdout
    assert "tap-visit numbers are scale diagnostics" in stdout


def test_mimo_long_signal_tutorial_smoke_small_scale(tmp_path: Path) -> None:
    stdout = _run_example(
        "examples/mimo_long_signal_state_space_stress.py",
        [
            "--samples",
            "512",
            "--batch",
            "1",
            "--inputs",
            "2",
            "--outputs",
            "2",
            "--full-order",
            "4",
            "--reduced-order",
            "2",
            "--markov-samples",
            "32",
            "--block-rows",
            "4",
            "--block-cols",
            "4",
            "--fir-taps",
            "256",
            "--repeats",
            "1",
        ],
        tmp_path,
    )
    assert "MIMO long-signal finite-Hankel/state-space stress" in stdout
    assert "finite MIMO block-Hankel reduction time" in stdout
    assert "scale diagnostics" in stdout


def test_public_claim_language_avoids_known_overclaim_phrases() -> None:
    """Guard a few phrases that previously made the public docs sound overbroad."""

    roots = [
        REPO_ROOT / name
        for name in ("README.md", "CHANGELOG.md", "docs", "examples", "lattice_dsp")
    ]
    forbidden = [
        "unique",
        "production-style",
        "complete AAK",
        "matrix AAK solver claim",
        "always fastest",
        "guaranteed fastest",
    ]
    hits: list[str] = []
    for root in roots:
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if not path.is_file() or path.suffix not in {".md", ".rst", ".py"}:
                continue
            if any(part in {"_build", "__pycache__"} for part in path.parts):
                continue
            text = path.read_text(encoding="utf-8")
            lower = text.lower()
            for phrase in forbidden:
                if phrase.lower() in lower:
                    hits.append(f"{path.relative_to(REPO_ROOT)}: {phrase}")
    assert hits == []
