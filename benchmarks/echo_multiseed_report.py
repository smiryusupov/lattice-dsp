"""Report generator for ``echo_multiseed_benchmark.py`` results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(missing_input_message(path))
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "scenarios" not in payload:
        raise ValueError("input JSON does not look like an echo multi-seed benchmark payload")
    return payload


def missing_input_message(path: Path) -> str:
    return (
        f"Input echo multi-seed benchmark JSON not found: {path}\n"
        "Generate it first, for example:\n"
        "python benchmarks/echo_multiseed_benchmark.py \\\n"
        "  --seeds 100 101 102 103 104 \\\n"
        "  --nonlinear-strengths 0.0 0.04 0.08 0.16 \\\n"
        "  --noise-snr-dbs 40 30 20 \\\n"
        "  --near-end-power-ratios 0.0 0.02 \\\n"
        "  --output echo-multiseed-benchmark.json"
    )


def scenario_row(item: dict[str, Any]) -> dict[str, Any]:
    scenario = item["scenario"]
    comparisons = item["comparisons"]
    keys = item["key_results"]

    def cmp_value(name: str, metric: str) -> Any:
        return comparisons.get(name, {}).get(metric, float("nan"))

    return {
        "nonlinearity": scenario["nonlinearity"],
        "nonlinear_strength": scenario["nonlinear_strength"],
        "noise_snr_db": scenario["noise_snr_db"],
        "near_end_power_ratio": scenario["near_end_power_ratio"],
        "fir_erle_db_median": keys["fir_erle_db_median"],
        "lattice_erle_db_median": keys["lattice_erle_db_median"],
        "hybrid_erle_db_median": keys["hybrid_erle_db_median"],
        "spectral_hybrid_erle_db_median": keys.get("spectral_hybrid_erle_db_median", float("nan")),
        "fir_runtime_s_median": keys.get("fir_runtime_s_median", float("nan")),
        "lattice_runtime_s_median": keys.get("lattice_runtime_s_median", float("nan")),
        "hybrid_runtime_s_median": keys.get("hybrid_runtime_s_median", float("nan")),
        "spectral_hybrid_runtime_s_median": keys.get(
            "spectral_hybrid_runtime_s_median", float("nan")
        ),
        "lattice_vs_fir_erle_gain_db_median": cmp_value("lattice_vs_fir", "erle_gain_db_median"),
        "lattice_vs_fir_erle_gain_db_min": cmp_value("lattice_vs_fir", "erle_gain_db_min"),
        "lattice_vs_fir_runtime_speedup_median": cmp_value(
            "lattice_vs_fir", "runtime_speedup_median"
        ),
        "hybrid_vs_fir_erle_gain_db_median": cmp_value("hybrid_vs_fir", "erle_gain_db_median"),
        "hybrid_vs_fir_erle_gain_db_min": cmp_value("hybrid_vs_fir", "erle_gain_db_min"),
        "hybrid_vs_fir_runtime_speedup_median": cmp_value(
            "hybrid_vs_fir", "runtime_speedup_median"
        ),
        "hybrid_vs_lattice_erle_gain_db_median": cmp_value(
            "hybrid_vs_lattice", "erle_gain_db_median"
        ),
        "spectral_hybrid_vs_fir_erle_gain_db_median": cmp_value(
            "spectral_hybrid_vs_fir", "erle_gain_db_median"
        ),
        "spectral_hybrid_vs_fir_erle_gain_db_min": cmp_value(
            "spectral_hybrid_vs_fir", "erle_gain_db_min"
        ),
        "spectral_hybrid_vs_fir_runtime_speedup_median": cmp_value(
            "spectral_hybrid_vs_fir", "runtime_speedup_median"
        ),
        "spectral_hybrid_vs_lattice_erle_gain_db_median": cmp_value(
            "spectral_hybrid_vs_lattice", "erle_gain_db_median"
        ),
        "spectral_hybrid_vs_toy_hybrid_erle_gain_db_median": cmp_value(
            "spectral_hybrid_vs_toy_hybrid", "erle_gain_db_median"
        ),
        "best_by_median_erle": item["best_by_median_erle"]["name"],
    }


def build_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [scenario_row(item) for item in payload["scenarios"]]


def robust_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    def count_positive(key: str) -> int:
        return sum(1 for row in rows if float(row[key]) > 0.0)

    return {
        "n_scenarios": len(rows),
        "lattice_beats_fir_median_count": count_positive("lattice_vs_fir_erle_gain_db_median"),
        "lattice_beats_fir_worst_count": count_positive("lattice_vs_fir_erle_gain_db_min"),
        "hybrid_beats_fir_median_count": count_positive("hybrid_vs_fir_erle_gain_db_median"),
        "hybrid_beats_fir_worst_count": count_positive("hybrid_vs_fir_erle_gain_db_min"),
        "hybrid_beats_lattice_median_count": count_positive(
            "hybrid_vs_lattice_erle_gain_db_median"
        ),
        "spectral_hybrid_beats_fir_median_count": count_positive(
            "spectral_hybrid_vs_fir_erle_gain_db_median"
        ),
        "spectral_hybrid_beats_fir_worst_count": count_positive(
            "spectral_hybrid_vs_fir_erle_gain_db_min"
        ),
        "spectral_hybrid_beats_lattice_median_count": count_positive(
            "spectral_hybrid_vs_lattice_erle_gain_db_median"
        ),
        "spectral_hybrid_beats_toy_hybrid_median_count": count_positive(
            "spectral_hybrid_vs_toy_hybrid_erle_gain_db_median"
        ),
        "min_lattice_vs_fir_gain_db": min(
            float(row["lattice_vs_fir_erle_gain_db_min"]) for row in rows
        ),
        "min_hybrid_vs_fir_gain_db": min(
            float(row["hybrid_vs_fir_erle_gain_db_min"]) for row in rows
        ),
        "median_lattice_vs_fir_speedup": sorted(
            float(row["lattice_vs_fir_runtime_speedup_median"]) for row in rows
        )[len(rows) // 2],
        "median_hybrid_vs_fir_speedup": sorted(
            float(row["hybrid_vs_fir_runtime_speedup_median"]) for row in rows
        )[len(rows) // 2],
        "median_spectral_hybrid_vs_fir_speedup": sorted(
            float(row["spectral_hybrid_vs_fir_runtime_speedup_median"]) for row in rows
        )[len(rows) // 2],
    }


def format_float(value: Any, digits: int = 4) -> str:
    value = float(value)
    if abs(value) >= 1000 or (abs(value) < 0.001 and value != 0):
        return f"{value:.3e}"
    return f"{value:.{digits}g}"


def render_markdown(payload: dict[str, Any], input_path: Path) -> str:
    metadata = payload.get("metadata", {})
    rows = build_rows(payload)
    summary = robust_summary(rows)
    lines: list[str] = []
    lines.append("# Echo/noise cancellation multi-seed benchmark report")
    lines.append("")
    lines.append(f"Input: `{input_path}`")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    for key in [
        "samples",
        "sample_rate",
        "seeds",
        "repeats",
        "nonlinearity",
        "nonlinear_strengths",
        "noise_snr_dbs",
        "near_end_power_ratios",
        "reflection_update_period",
        "scale_reflection_mu_by_period",
        "iir_order",
        "fir_order",
    ]:
        if key in metadata:
            lines.append(f"- {key}: `{metadata[key]}`")
    lines.append("")
    lines.append("## Robust summary")
    lines.append("")
    if summary:
        lines.append(
            f"Across `{summary['n_scenarios']}` scenario(s), lattice/IIR beat FIR/NLMS in median ERLE "
            f"for `{summary['lattice_beats_fir_median_count']}` scenario(s) and in every-seed/worst-case ERLE "
            f"for `{summary['lattice_beats_fir_worst_count']}` scenario(s)."
        )
        lines.append(
            f"The hybrid method beat FIR/NLMS in median ERLE for `{summary['hybrid_beats_fir_median_count']}` "
            f"scenario(s) and in every-seed/worst-case ERLE for `{summary['hybrid_beats_fir_worst_count']}` scenario(s)."
        )
        lines.append(
            f"Median lattice-vs-FIR runtime speedup across scenarios: "
            f"`{format_float(summary['median_lattice_vs_fir_speedup'])}×`; "
            f"median toy-hybrid-vs-FIR runtime speedup: "
            f"`{format_float(summary['median_hybrid_vs_fir_speedup'])}×`; "
            f"median spectral-hybrid-vs-FIR runtime speedup: "
            f"`{format_float(summary['median_spectral_hybrid_vs_fir_speedup'])}×`."
        )
        lines.append(
            f"The spectral hybrid beat FIR/NLMS in median ERLE for "
            f"`{summary['spectral_hybrid_beats_fir_median_count']}` scenario(s), "
            f"beat lattice-only for `{summary['spectral_hybrid_beats_lattice_median_count']}` scenario(s), "
            f"and beat the toy hybrid for `{summary['spectral_hybrid_beats_toy_hybrid_median_count']}` scenario(s)."
        )
    else:
        lines.append("No scenario rows were available.")
    lines.append("")
    lines.append("## Scenario table")
    lines.append("")
    headers = [
        "nonlin.",
        "noise SNR",
        "near",
        "FIR ERLE",
        "lattice ERLE",
        "hybrid ERLE",
        "spectral ERLE",
        "lat-FIR gain",
        "lat-FIR worst",
        "lat speedup",
        "hyb-FIR gain",
        "hyb-FIR worst",
        "hyb speedup",
        "hyb-lat gain",
        "spec-FIR gain",
        "spec speedup",
        "spec-lat gain",
        "spec-toy gain",
        "best",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---:"] * (len(headers) - 1) + ["---"]) + "|")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    format_float(row["nonlinear_strength"]),
                    format_float(row["noise_snr_db"]),
                    format_float(row["near_end_power_ratio"]),
                    format_float(row["fir_erle_db_median"]),
                    format_float(row["lattice_erle_db_median"]),
                    format_float(row["hybrid_erle_db_median"]),
                    format_float(row["spectral_hybrid_erle_db_median"]),
                    format_float(row["lattice_vs_fir_erle_gain_db_median"]),
                    format_float(row["lattice_vs_fir_erle_gain_db_min"]),
                    format_float(row["lattice_vs_fir_runtime_speedup_median"]),
                    format_float(row["hybrid_vs_fir_erle_gain_db_median"]),
                    format_float(row["hybrid_vs_fir_erle_gain_db_min"]),
                    format_float(row["hybrid_vs_fir_runtime_speedup_median"]),
                    format_float(row["hybrid_vs_lattice_erle_gain_db_median"]),
                    format_float(row["spectral_hybrid_vs_fir_erle_gain_db_median"]),
                    format_float(row["spectral_hybrid_vs_fir_runtime_speedup_median"]),
                    format_float(row["spectral_hybrid_vs_lattice_erle_gain_db_median"]),
                    format_float(row["spectral_hybrid_vs_toy_hybrid_erle_gain_db_median"]),
                    str(row["best_by_median_erle"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Use this report to separate the linear-model claim from the residual-model claim: "
        "`lattice_iir_only` measures the stable adaptive-IIR stage, while "
        "`lattice_iir_plus_toy_residual_suppressor` is a fixed-gain sanity check, while "
        "`lattice_iir_plus_spectral_residual_suppressor` is the first deterministic residual-model baseline. "
        "Keep the residual processor interface dependency-free; stronger downstream processors can be evaluated outside the core package."
    )
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--markdown-output", type=Path, default=Path("reports/echo-multiseed-report.md")
    )
    parser.add_argument(
        "--csv-output", type=Path, default=Path("reports/echo-multiseed-summary.csv")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        payload = load_payload(args.input)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    rows = build_rows(payload)
    markdown = render_markdown(payload, args.input)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown, encoding="utf-8")
    write_csv(rows, args.csv_output)
    print(markdown)
    print(f"Wrote {args.markdown_output}")
    print(f"Wrote {args.csv_output}")


if __name__ == "__main__":
    main()
