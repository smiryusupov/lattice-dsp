"""Summarize multi-seed adaptive reflection-period sweeps.

This helper reads JSON produced by ``benchmarks/adaptive_multiseed_sweep.py``
and recommends a robust ``reflection_update_period``.  Unlike the single-seed
report, it can constrain both median quality and worst-case quality across
seeds.

Example
-------
python benchmarks/adaptive_multiseed_report.py adaptive-multiseed-sweep.json \
    --markdown-output adaptive-multiseed-report.md \
    --csv-output adaptive-multiseed-pareto.csv \
    --max-tail-mse-ratio-median 1.5 \
    --max-tail-mse-ratio-worst 2.0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


AGGREGATE_COLUMNS = [
    "reflection_update_period",
    "n_seeds",
    "median_s_median",
    "speedup_vs_period1_median",
    "speedup_vs_period1_min",
    "mse_tail_median",
    "mse_tail_max",
    "tail_mse_ratio_median",
    "tail_mse_ratio_p90",
    "tail_mse_ratio_max",
    "reflection_l2_error_median",
    "reflection_l2_error_max",
    "taps_l2_error_median",
    "taps_l2_error_max",
    "stability_margin_median",
    "stability_margin_min",
]


def _as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def candidate_multiseed_files(path: Path) -> list[Path]:
    search_dir = path.parent if path.parent != Path("") else Path.cwd()
    patterns = [
        "adaptive-multiseed*.json",
        "*multiseed*sweep*.json",
        "adaptive-period*.json",
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for pattern in patterns:
        for candidate in sorted(search_dir.glob(pattern)):
            if candidate.is_file() and candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
    return out


def missing_input_message(path: Path) -> str:
    lines = [
        f"Input multi-seed sweep JSON not found: {path}",
        f"Current working directory: {Path.cwd()}",
        "",
        "Create one with:",
        "  python benchmarks/adaptive_multiseed_sweep.py \\",
        "    --seeds 100 101 102 103 104 \\",
        "    --periods 1 2 4 8 16 32 \\",
        "    --samples 20000 \\",
        "    --repeats 3 \\",
        "    --output adaptive-multiseed-sweep.json \\",
        "    --csv-output adaptive-multiseed-aggregate.csv",
    ]
    candidates = candidate_multiseed_files(path)
    if candidates:
        lines.extend(["", "Possible sweep files found nearby:"])
        lines.extend(f"  - {candidate}" for candidate in candidates[:10])
    return "\n".join(lines)


def load_multiseed(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(missing_input_message(path))
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("aggregate_results")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} does not look like a multi-seed adaptive-period sweep JSON file")
    return payload


def format_float(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if abs(number) >= 1000 or (0 < abs(number) < 1e-3):
        return f"{number:.{digits}e}"
    return f"{number:.{digits}g}"


def robust_pareto_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows not dominated by speed and worst-case tail-MSE ratio."""
    out: list[dict[str, Any]] = []
    for row in rows:
        t = _as_float(row["median_s_median"])
        q = _as_float(row["tail_mse_ratio_max"])
        dominated = False
        for other in rows:
            if other is row:
                continue
            ot = _as_float(other["median_s_median"])
            oq = _as_float(other["tail_mse_ratio_max"])
            if ot <= t and oq <= q and (ot < t or oq < q):
                dominated = True
                break
        if not dominated:
            out.append(row)
    return sorted(out, key=lambda r: _as_float(r["median_s_median"]))


def recommend_period(
    rows: list[dict[str, Any]],
    *,
    max_tail_mse_ratio_median: float,
    max_tail_mse_ratio_worst: float,
    min_stability_margin: float,
    prefer: str,
) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if _as_float(row.get("tail_mse_ratio_median")) <= max_tail_mse_ratio_median
        and _as_float(row.get("tail_mse_ratio_max")) <= max_tail_mse_ratio_worst
        and _as_float(row.get("stability_margin_min")) >= min_stability_margin
    ]
    if not candidates:
        candidates = rows
        prefer = "lowest-worst-tail-mse"

    if prefer == "fastest":
        return min(candidates, key=lambda r: _as_float(r["median_s_median"]))
    if prefer == "lowest-median-tail-mse":
        return min(candidates, key=lambda r: _as_float(r["tail_mse_ratio_median"]))
    if prefer == "lowest-worst-tail-mse":
        return min(candidates, key=lambda r: _as_float(r["tail_mse_ratio_max"]))
    raise ValueError("invalid preference")


def markdown_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| period | seeds | median s | med speedup | min speedup | "
        "med tail ratio | p90 tail ratio | worst tail ratio | "
        "med refl. err | worst refl. err | min stability |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "| {period} | {seeds} | {median_s} | {speedup} | {speedup_min} | {ratio_med} | "
            "{ratio_p90} | {ratio_max} | {refl_med} | {refl_max} | {margin_min} |".format(
                period=row["reflection_update_period"],
                seeds=row.get("n_seeds", "n/a"),
                median_s=format_float(row.get("median_s_median")),
                speedup=format_float(row.get("speedup_vs_period1_median")),
                speedup_min=format_float(row.get("speedup_vs_period1_min")),
                ratio_med=format_float(row.get("tail_mse_ratio_median")),
                ratio_p90=format_float(row.get("tail_mse_ratio_p90")),
                ratio_max=format_float(row.get("tail_mse_ratio_max")),
                refl_med=format_float(row.get("reflection_l2_error_median")),
                refl_max=format_float(row.get("reflection_l2_error_max")),
                margin_min=format_float(row.get("stability_margin_min")),
            )
        )
    return "\n".join(lines)


def build_report(
    payload: dict[str, Any],
    *,
    input_path: Path,
    max_tail_mse_ratio_median: float,
    max_tail_mse_ratio_worst: float,
    min_stability_margin: float,
    prefer: str,
) -> str:
    rows = payload["aggregate_results"]
    metadata = payload.get("metadata", {})
    recommended = recommend_period(
        rows,
        max_tail_mse_ratio_median=max_tail_mse_ratio_median,
        max_tail_mse_ratio_worst=max_tail_mse_ratio_worst,
        min_stability_margin=min_stability_margin,
        prefer=prefer,
    )
    pareto = robust_pareto_rows(rows)

    seeds = metadata.get("seeds", [])
    lines = [
        "# Multi-seed adaptive reflection-update period report",
        "",
        f"Input: `{input_path}`",
        "",
        "## Metadata",
        "",
        f"- samples: `{metadata.get('samples', 'n/a')}`",
        f"- repeats per seed/period: `{metadata.get('repeats', 'n/a')}`",
        f"- seeds: `{seeds}`",
        f"- scale_reflection_mu_by_period: `{metadata.get('scale_reflection_mu_by_period', 'n/a')}`",
        f"- mu_reflection: `{metadata.get('mu_reflection', 'n/a')}`",
        f"- mu_taps: `{metadata.get('mu_taps', 'n/a')}`",
        "",
        "## Robust recommended period",
        "",
        (
            f"Recommended `reflection_update_period={recommended['reflection_update_period']}` using `{prefer}` "
            f"with median tail-MSE ratio <= `{max_tail_mse_ratio_median}`, worst-case tail-MSE ratio <= "
            f"`{max_tail_mse_ratio_worst}`, and minimum stability margin >= `{min_stability_margin}`."
        ),
        "",
        "This point has "
        f"median speedup `{format_float(recommended.get('speedup_vs_period1_median'))}×`, "
        f"minimum speedup `{format_float(recommended.get('speedup_vs_period1_min'))}×`, "
        f"median tail-MSE ratio `{format_float(recommended.get('tail_mse_ratio_median'))}`, "
        f"worst-case tail-MSE ratio `{format_float(recommended.get('tail_mse_ratio_max'))}`, and "
        f"minimum stability margin `{format_float(recommended.get('stability_margin_min'))}`.",
        "",
        "## Aggregate table",
        "",
        markdown_table(rows),
        "",
        "## Robust Pareto frontier",
        "",
        "The frontier uses median runtime and worst-case tail-MSE ratio across seeds.",
        "",
        markdown_table(pareto),
        "",
    ]
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AGGREGATE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in AGGREGATE_COLUMNS})


def maybe_write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency branch
        raise RuntimeError("matplotlib is required for --plot-output") from exc

    periods = [int(row["reflection_update_period"]) for row in rows]
    speedups = [_as_float(row.get("speedup_vs_period1_median")) for row in rows]
    worst_ratios = [_as_float(row.get("tail_mse_ratio_max")) for row in rows]
    median_ratios = [_as_float(row.get("tail_mse_ratio_median")) for row in rows]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(periods, speedups, marker="o", label="median speedup")
    ax1.set_xlabel("reflection_update_period")
    ax1.set_ylabel("median speedup")
    ax2 = ax1.twinx()
    ax2.plot(periods, median_ratios, marker="s", linestyle="--", label="median tail-MSE ratio")
    ax2.plot(periods, worst_ratios, marker="^", linestyle=":", label="worst tail-MSE ratio")
    ax2.set_ylabel("tail-MSE ratio")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON file from adaptive_multiseed_sweep.py")
    parser.add_argument(
        "--markdown-output", type=Path, default=Path("reports/adaptive-multiseed-report.md")
    )
    parser.add_argument(
        "--csv-output", type=Path, default=None, help="Optional CSV containing robust Pareto rows"
    )
    parser.add_argument(
        "--plot-output", type=Path, default=None, help="Optional PNG plot; requires matplotlib"
    )
    parser.add_argument("--max-tail-mse-ratio-median", type=float, default=1.5)
    parser.add_argument("--max-tail-mse-ratio-worst", type=float, default=2.0)
    parser.add_argument("--min-stability-margin", type=float, default=0.05)
    parser.add_argument(
        "--prefer",
        choices=["fastest", "lowest-median-tail-mse", "lowest-worst-tail-mse"],
        default="fastest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        payload = load_multiseed(args.input)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None
    rows = payload["aggregate_results"]
    report = build_report(
        payload,
        input_path=args.input,
        max_tail_mse_ratio_median=args.max_tail_mse_ratio_median,
        max_tail_mse_ratio_worst=args.max_tail_mse_ratio_worst,
        min_stability_margin=args.min_stability_margin,
        prefer=args.prefer,
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {args.markdown_output}")

    if args.csv_output is not None:
        write_csv(args.csv_output, robust_pareto_rows(rows))
        print(f"Wrote {args.csv_output}")

    if args.plot_output is not None:
        maybe_write_plot(args.plot_output, rows)
        print(f"Wrote {args.plot_output}")


if __name__ == "__main__":
    main()
