"""Summarize adaptive-period sweep JSON files.

The sweep benchmark measures the speed/quality trade-off of decimating
reflection/denominator updates.  This helper turns the raw JSON produced by
``benchmarks/adaptive_period_sweep.py`` into a small Markdown report and an
optional Pareto CSV.  It is intentionally dependency-light: plotting is skipped
unless matplotlib is installed and ``--plot-output`` is requested.

Example
-------
python benchmarks/adaptive_period_report.py adaptive-period-sweep-scaled.json \
    --markdown-output adaptive-period-report.md \
    --csv-output adaptive-period-pareto.csv \
    --max-tail-mse-ratio 1.5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


NUMERIC_COLUMNS = [
    "reflection_update_period",
    "median_s",
    "speedup_vs_first_period",
    "mse_tail",
    "tail_mse_ratio_vs_first_period",
    "mse_total",
    "reflection_l2_error",
    "taps_l2_error",
    "stability_margin",
]


def candidate_sweep_files(path: Path) -> list[Path]:
    """Return likely sweep JSON files near ``path`` for better error messages."""
    search_dir = path.parent if path.parent != Path("") else Path.cwd()
    if not search_dir.exists():
        search_dir = Path.cwd()
    patterns = [
        "adaptive-period-sweep*.json",
        "*adaptive*period*sweep*.json",
        "*sweep*.json",
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
    """Build a helpful FileNotFound message for common sweep/report mixups."""
    lines = [
        f"Input sweep JSON not found: {path}",
        f"Current working directory: {Path.cwd()}",
        "",
        "Create the scaled sweep file with:",
        "  python benchmarks/adaptive_period_sweep.py \\",
        "    --periods 1 2 4 8 16 32 \\",
        "    --samples 20000 \\",
        "    --repeats 5 \\",
        "    --output adaptive-period-sweep-scaled.json \\",
        "    --csv-output adaptive-period-sweep-scaled.csv",
        "",
        "Or pass an existing sweep file, for example:",
        "  python benchmarks/adaptive_period_report.py adaptive-period-sweep-fixed.json",
    ]
    candidates = candidate_sweep_files(path)
    if candidates:
        lines.extend(["", "Possible sweep files found nearby:"])
        lines.extend(f"  - {candidate}" for candidate in candidates[:10])
    return "\n".join(lines)


def _as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def load_sweep(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(missing_input_message(path))
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "results" not in payload or not isinstance(payload["results"], list):
        raise ValueError(f"{path} does not look like an adaptive-period sweep JSON file")
    if not payload["results"]:
        raise ValueError(f"{path} contains no sweep results")
    return payload


def pareto_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows that are not dominated in both speed and tail MSE.

    A row is dominated when another row is at least as fast and has at least as
    low tail MSE, with one of those two metrics strictly better.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        t = _as_float(row["median_s"])
        q = _as_float(row["mse_tail"])
        dominated = False
        for other in rows:
            if other is row:
                continue
            ot = _as_float(other["median_s"])
            oq = _as_float(other["mse_tail"])
            if ot <= t and oq <= q and (ot < t or oq < q):
                dominated = True
                break
        if not dominated:
            out.append(row)
    return sorted(out, key=lambda r: _as_float(r["median_s"]))


def recommend_period(
    rows: list[dict[str, Any]],
    *,
    max_tail_mse_ratio: float,
    min_stability_margin: float,
    prefer: str,
) -> dict[str, Any]:
    """Choose a period from a sweep.

    ``prefer='fastest'`` selects the fastest row satisfying quality constraints.
    ``prefer='lowest-tail-mse'`` selects the best tail MSE among constrained rows.
    """
    candidates = [
        row
        for row in rows
        if _as_float(row.get("tail_mse_ratio_vs_first_period")) <= max_tail_mse_ratio
        and _as_float(row.get("stability_margin")) >= min_stability_margin
    ]
    if not candidates:
        # Fall back to the best quality result.  The caller can inspect the
        # returned row and see that it does not satisfy the requested threshold.
        candidates = rows
        prefer = "lowest-tail-mse"

    if prefer == "fastest":
        return min(candidates, key=lambda r: _as_float(r["median_s"]))
    if prefer == "lowest-tail-mse":
        return min(candidates, key=lambda r: _as_float(r["mse_tail"]))
    raise ValueError("prefer must be 'fastest' or 'lowest-tail-mse'")


def format_float(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if abs(number) >= 1000 or (0 < abs(number) < 1e-3):
        return f"{number:.{digits}e}"
    return f"{number:.{digits}g}"


def markdown_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| period | median s | speedup | tail MSE | tail MSE ratio | "
        "refl. L2 err | taps L2 err | stability margin |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "| {period} | {median} | {speedup} | {tail} | {ratio} | {refl} | {taps} | {margin} |".format(
                period=row["reflection_update_period"],
                median=format_float(row["median_s"]),
                speedup=format_float(row.get("speedup_vs_first_period")),
                tail=format_float(row["mse_tail"]),
                ratio=format_float(row.get("tail_mse_ratio_vs_first_period")),
                refl=format_float(row["reflection_l2_error"]),
                taps=format_float(row["taps_l2_error"]),
                margin=format_float(row["stability_margin"]),
            )
        )
    return "\n".join(lines)


def build_report(
    payload: dict[str, Any],
    *,
    input_path: Path,
    max_tail_mse_ratio: float,
    min_stability_margin: float,
    prefer: str,
) -> str:
    rows = payload["results"]
    metadata = payload.get("metadata", {})
    recommended = recommend_period(
        rows,
        max_tail_mse_ratio=max_tail_mse_ratio,
        min_stability_margin=min_stability_margin,
        prefer=prefer,
    )
    pareto = pareto_rows(rows)

    lines = [
        "# Adaptive reflection-update period report",
        "",
        f"Input: `{input_path}`",
        "",
        "## Metadata",
        "",
        f"- samples: `{metadata.get('samples', 'n/a')}`",
        f"- repeats: `{metadata.get('repeats', 'n/a')}`",
        f"- scale_reflection_mu_by_period: `{metadata.get('scale_reflection_mu_by_period', 'n/a')}`",
        f"- mu_reflection: `{metadata.get('mu_reflection', 'n/a')}`",
        f"- mu_taps: `{metadata.get('mu_taps', 'n/a')}`",
        "",
        "## Recommended period",
        "",
        (
            f"Recommended `reflection_update_period={recommended['reflection_update_period']}` "
            f"using `{prefer}` with `max_tail_mse_ratio={max_tail_mse_ratio}` and "
            f"`min_stability_margin={min_stability_margin}`."
        ),
        "",
        "This point has "
        f"`{format_float(recommended.get('speedup_vs_first_period'))}×` speedup, "
        f"tail-MSE ratio `{format_float(recommended.get('tail_mse_ratio_vs_first_period'))}`, "
        f"reflection L2 error `{format_float(recommended.get('reflection_l2_error'))}`, and "
        f"stability margin `{format_float(recommended.get('stability_margin'))}`.",
        "",
        "## Sweep table",
        "",
        markdown_table(rows),
        "",
        "## Pareto frontier",
        "",
        markdown_table(pareto),
        "",
    ]
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=NUMERIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in NUMERIC_COLUMNS})


def maybe_write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency branch
        raise RuntimeError("matplotlib is required for --plot-output") from exc

    periods = [int(row["reflection_update_period"]) for row in rows]
    speedups = [_as_float(row.get("speedup_vs_first_period")) for row in rows]
    ratios = [_as_float(row.get("tail_mse_ratio_vs_first_period")) for row in rows]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(periods, speedups, marker="o", label="speedup vs period 1")
    ax1.set_xlabel("reflection_update_period")
    ax1.set_ylabel("speedup")
    ax2 = ax1.twinx()
    ax2.plot(periods, ratios, marker="s", linestyle="--", label="tail-MSE ratio")
    ax2.set_ylabel("tail-MSE ratio")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON file from adaptive_period_sweep.py")
    parser.add_argument(
        "--markdown-output", type=Path, default=Path("reports/adaptive-period-report.md")
    )
    parser.add_argument(
        "--csv-output", type=Path, default=None, help="Optional CSV containing the Pareto frontier"
    )
    parser.add_argument(
        "--plot-output", type=Path, default=None, help="Optional PNG plot; requires matplotlib"
    )
    parser.add_argument("--max-tail-mse-ratio", type=float, default=1.5)
    parser.add_argument("--min-stability-margin", type=float, default=0.05)
    parser.add_argument("--prefer", choices=["fastest", "lowest-tail-mse"], default="fastest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        payload = load_sweep(args.input)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None
    rows = payload["results"]
    report = build_report(
        payload,
        input_path=args.input,
        max_tail_mse_ratio=args.max_tail_mse_ratio,
        min_stability_margin=args.min_stability_margin,
        prefer=args.prefer,
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {args.markdown_output}")

    if args.csv_output is not None:
        write_csv(args.csv_output, pareto_rows(rows))
        print(f"Wrote {args.csv_output}")

    if args.plot_output is not None:
        maybe_write_plot(args.plot_output, rows)
        print(f"Wrote {args.plot_output}")


if __name__ == "__main__":
    main()
