"""Create compact benchmark figures from generated JSON/CSV artifacts.

The benchmark scripts intentionally keep their numeric payloads machine-readable.
This helper turns the same payloads into a few human-readable plots for the
Sphinx benchmark gallery.  It is deliberately best-effort: unsupported payloads
are left as downloadable data files rather than failing the docs build.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from collections.abc import Iterable
from typing import Any

IMAGE_DPI = 150

# Keys ordered by how useful they usually are in the public benchmark pages.
SPEED_KEYS = (
    "speedup_vs_full",
    "filter_speedup",
    "process_speedup",
    "speedup",
    "speedup_direct_over_levinson",
    "one_shot_end_to_end_speedup",
    "amortized_end_to_end_speedup",
    "static_gain_improvement",
    "speedup_vs_first_period",
    "speedup_scalar_blocks_vs_full_mimo",
)
TIME_KEYS = (
    "median_s",
    "compiled_s",
    "python_s",
    "full_filter_median_s",
    "reduced_filter_median_s",
    "full_process_median_s",
    "reduced_process_median_s",
    "reduction_time_s",
    "reduce_s",
    "realize_s",
    "time_s",
    "direct_dense_seconds_median",
    "block_levinson_seconds_median",
    "pick_build_s",
    "pick_eig_s",
    "psd_check_s",
    "constant_solution_s",
    "potapov_build_s",
    "potapov_eval_s",
    "j_check_s",
    "full_mimo_pick_s",
    "scalar_blocks_pick_s",
    "scalar_block_time_s",
)
ERROR_KEYS = (
    "relative_output_error",
    "relative_markov_error",
    "static_gain_relative_error",
    "state_response_relative_error",
    "relative_impulse_error",
    "rel_mse",
    "rel_mse_on_random_batch",
    "tail_mse_ratio_vs_first_period",
    "relative_tail_error",
    "coefficient_difference_norm",
    "unitarity_error",
    "relative_difference",
    "polar_factor_relative_error",
    "hankelized_hankel_error",
    "unconstrained_hankel_error",
    "sigma_next",
    "max_tangential_residual",
    "constant_solution_relative_error",
    "j_inner_residual",
    "diagonal_block_relative_error",
    "diagonal_block_eigenvalue_relative_error",
)
QUALITY_KEYS = (
    "snr_db",
    "snr_db_on_random_batch",
    "output_snr_db",
    "erle_db",
    "retained_hankel_energy",
    "max_reflection_spectral_norm",
    "max_reflection_singular_value",
    "max_pole_radius",
    "min_pick_eigenvalue",
    "constant_solution_sigma_max",
)


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _load_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _rows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize the benchmark payload conventions used in ``benchmarks/``."""

    for key in ("rows", "results", "cases"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, dict)]

    benchmarks = payload.get("benchmarks")
    if isinstance(benchmarks, dict):
        rows: list[dict[str, Any]] = []
        for name, values in benchmarks.items():
            row: dict[str, Any] = {"benchmark": name}
            if isinstance(values, dict):
                row.update(values)
            else:
                row["value"] = values
            rows.append(row)
        return rows

    # Some small validation benchmarks write a single flat dictionary.
    flat_numeric = {key: value for key, value in payload.items() if _to_float(value) is not None}
    if flat_numeric:
        row = dict(payload)
        row.setdefault("case", "result")
        return [row]
    return []


def _rows_from_artifact(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = _load_json(path)
        return _rows_from_payload(payload) if payload is not None else []
    if path.suffix.lower() == ".csv":
        return _load_csv(path)
    return []


def _shorten(text: str, max_len: int = 28) -> str:
    text = str(text).replace("_", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _row_label(row: dict[str, Any], index: int) -> str:
    parts: list[str] = []
    for key in ("benchmark", "method", "case", "name", "diagnostic_classification"):
        value = row.get(key)
        if value not in (None, ""):
            parts.append(str(value))
            break
    for key in (
        "full_order",
        "reduced_order",
        "target_order",
        "order",
        "rank",
        "lattice_order",
        "dim",
        "reflection_update_period",
    ):
        value = row.get(key)
        if value not in (None, ""):
            short_key = {
                "full_order": "full",
                "reduced_order": "red",
                "target_order": "target",
                "lattice_order": "lat",
                "reflection_update_period": "period",
            }.get(key, key)
            parts.append(f"{short_key}={value}")
    if not parts:
        parts.append(f"row {index + 1}")
    return _shorten(" / ".join(parts))


def _available_metrics(rows: list[dict[str, Any]], candidates: Iterable[str]) -> list[str]:
    found: list[str] = []
    for key in candidates:
        values = [_to_float(row.get(key)) for row in rows]
        if any(value is not None for value in values):
            found.append(key)
    return found


def _metric_values(rows: list[dict[str, Any]], key: str) -> list[float | None]:
    return [_to_float(row.get(key)) for row in rows]


def _sanitize_filename(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text.lower())
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "metric"


def _plot_grouped_metrics(
    rows: list[dict[str, Any]],
    metrics: list[str],
    *,
    output: Path,
    title: str,
    ylabel: str,
    logy: bool = False,
) -> bool:
    if not rows or not metrics:
        return False

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:  # pragma: no cover - optional docs dependency.
        return False

    labels = [_row_label(row, idx) for idx, row in enumerate(rows)]
    x = np.arange(len(rows), dtype=float)
    width = min(0.8 / max(1, len(metrics)), 0.28)

    fig_width = max(7.0, min(15.0, 0.45 * len(rows) + 2.5))
    fig, ax = plt.subplots(figsize=(fig_width, 4.6))
    plotted = False
    for metric_index, metric in enumerate(metrics):
        values = _metric_values(rows, metric)
        numeric = [float("nan") if value is None else value for value in values]
        if all(math.isnan(value) for value in numeric):
            continue
        offset = (metric_index - (len(metrics) - 1) / 2.0) * width
        ax.bar(x + offset, numeric, width=width, label=metric.replace("_", " "))
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    if logy:
        positive_values = [
            value
            for metric in metrics
            for value in _metric_values(rows, metric)
            if value is not None and value > 0.0
        ]
        if positive_values:
            ax.set_yscale("log")
    if len(metrics) > 1:
        ax.legend(fontsize="small")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=IMAGE_DPI)
    plt.close(fig)
    return True


def _plot_benchmark_pair(rows: list[dict[str, Any]], *, output: Path, title: str) -> bool:
    """Special plot for payloads that compare two timing columns per row."""

    pair_candidates = [
        ("compiled_s", "python_s"),
        ("full_filter_median_s", "reduced_filter_median_s"),
        ("full_process_median_s", "reduced_process_median_s"),
        ("direct_dense_seconds_median", "block_levinson_seconds_median"),
    ]
    pair = next(
        (
            candidate
            for candidate in pair_candidates
            if all(any(_to_float(row.get(key)) is not None for row in rows) for key in candidate)
        ),
        None,
    )
    if pair is None:
        return False
    return _plot_grouped_metrics(
        rows,
        list(pair),
        output=output,
        title=title,
        ylabel="seconds",
        logy=True,
    )


def create_benchmark_visuals(
    artifact_dir: Path, *, slug: str | None = None, title: str | None = None
) -> list[Path]:
    """Create best-effort PNG summaries for benchmark JSON/CSV artifacts.

    Parameters
    ----------
    artifact_dir:
        Directory containing generated benchmark artifacts.
    slug, title:
        Optional names used in filenames and plot titles.

    Returns
    -------
    list[pathlib.Path]
        Paths of figures that were written.
    """

    if not artifact_dir.exists():
        return []

    # Prefer JSON because it carries the complete payload; fall back to CSV for
    # sweep helpers that also write tabular output.
    artifacts = sorted(artifact_dir.glob("*.json")) or sorted(artifact_dir.glob("*.csv"))
    if not artifacts:
        return []

    rows = _rows_from_artifact(artifacts[0])
    rows = [row for row in rows if isinstance(row, dict)]
    if not rows:
        return []

    base = _sanitize_filename(slug or artifacts[0].stem)
    title_prefix = title or (slug or artifacts[0].stem).replace("_", " ").title()
    written: list[Path] = []

    speed_metrics = _available_metrics(rows, SPEED_KEYS)[:3]
    if speed_metrics:
        out = artifact_dir / f"{base}_speedup_summary.png"
        if _plot_grouped_metrics(
            rows,
            speed_metrics,
            output=out,
            title=f"{title_prefix}: speedup metrics",
            ylabel="speedup ratio",
            logy=False,
        ):
            written.append(out)

    error_metrics = _available_metrics(rows, ERROR_KEYS)[:3]
    if error_metrics:
        out = artifact_dir / f"{base}_error_summary.png"
        if _plot_grouped_metrics(
            rows,
            error_metrics,
            output=out,
            title=f"{title_prefix}: error/validation metrics",
            ylabel="error or residual metric",
            logy=True,
        ):
            written.append(out)

    pair_out = artifact_dir / f"{base}_timing_comparison.png"
    if _plot_benchmark_pair(rows, output=pair_out, title=f"{title_prefix}: timing comparison"):
        written.append(pair_out)
    else:
        time_metrics = _available_metrics(rows, TIME_KEYS)[:3]
        if time_metrics:
            out = artifact_dir / f"{base}_runtime_summary.png"
            if _plot_grouped_metrics(
                rows,
                time_metrics,
                output=out,
                title=f"{title_prefix}: runtime metrics",
                ylabel="seconds",
                logy=True,
            ):
                written.append(out)

    quality_metrics = _available_metrics(rows, QUALITY_KEYS)[:3]
    if quality_metrics:
        out = artifact_dir / f"{base}_quality_summary.png"
        if _plot_grouped_metrics(
            rows,
            quality_metrics,
            output=out,
            title=f"{title_prefix}: quality/stability metrics",
            ylabel="quality/stability metric",
            logy=False,
        ):
            written.append(out)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create benchmark visual summaries from JSON/CSV artifacts."
    )
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--slug", default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    for path in create_benchmark_visuals(args.artifact_dir, slug=args.slug, title=args.title):
        print(path.as_posix())


if __name__ == "__main__":
    main()
