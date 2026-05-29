from pathlib import Path

import pytest

from benchmarks.echo_multiseed_benchmark import (
    aggregate_case_rows,
    compare_case_pairs,
    parse_float_values,
    parse_int_values,
    write_scenario_csv,
)
from benchmarks.echo_multiseed_report import build_rows, missing_input_message, robust_summary


def test_parse_value_helpers_accept_commas_and_spaces():
    assert parse_int_values(["1,2", "3", "2"]) == [1, 2, 3]
    assert parse_float_values(["0.0,0.1", "0.2", "0.1"]) == [0.0, 0.1, 0.2]


def test_aggregate_case_rows_tracks_erle_and_runtime():
    rows = [
        {
            "seed": 1,
            "name": "lattice_iir_only",
            "erle_db": 20.0,
            "segmental_erle_median_db": 21.0,
            "mse_improvement_db": 20.0,
            "output_mse": 0.01,
            "residual_power_db": -20.0,
            "median_s": 0.1,
        },
        {
            "seed": 2,
            "name": "lattice_iir_only",
            "erle_db": 22.0,
            "segmental_erle_median_db": 23.0,
            "mse_improvement_db": 22.0,
            "output_mse": 0.02,
            "residual_power_db": -18.0,
            "median_s": 0.2,
        },
    ]
    aggregate = aggregate_case_rows(rows)
    assert aggregate[0]["name"] == "lattice_iir_only"
    assert aggregate[0]["erle_db_median"] == 21.0
    assert aggregate[0]["erle_db_min"] == 20.0
    assert aggregate[0]["median_s_median"] == pytest.approx(0.15)


def test_compare_case_pairs_computes_gain_and_speedup():
    rows = [
        {
            "seed": 1,
            "name": "lattice",
            "erle_db": 20.0,
            "mse_improvement_db": 20.0,
            "median_s": 0.1,
        },
        {"seed": 1, "name": "fir", "erle_db": 18.0, "mse_improvement_db": 18.0, "median_s": 0.5},
        {
            "seed": 2,
            "name": "lattice",
            "erle_db": 21.0,
            "mse_improvement_db": 21.0,
            "median_s": 0.2,
        },
        {"seed": 2, "name": "fir", "erle_db": 20.0, "mse_improvement_db": 20.0, "median_s": 0.4},
    ]
    comparison = compare_case_pairs(rows, "lattice", "fir")
    assert comparison["erle_gain_db_median"] == 1.5
    assert comparison["erle_gain_db_min"] == 1.0
    assert comparison["runtime_speedup_median"] == 3.5


def test_report_rows_and_summary_from_payload(tmp_path: Path):
    payload = {
        "metadata": {"samples": 128},
        "scenarios": [
            {
                "scenario": {
                    "nonlinearity": "tanh",
                    "nonlinear_strength": 0.08,
                    "noise_snr_db": 30.0,
                    "near_end_power_ratio": 0.02,
                },
                "key_results": {
                    "fir_erle_db_median": 18.0,
                    "lattice_erle_db_median": 20.0,
                    "hybrid_erle_db_median": 21.0,
                    "lattice_runtime_s_median": 0.1,
                    "fir_runtime_s_median": 0.5,
                    "hybrid_runtime_s_median": 0.1,
                },
                "comparisons": {
                    "lattice_vs_fir": {
                        "erle_gain_db_median": 2.0,
                        "erle_gain_db_min": 1.5,
                        "runtime_speedup_median": 5.0,
                    },
                    "hybrid_vs_fir": {
                        "erle_gain_db_median": 3.0,
                        "erle_gain_db_min": 2.5,
                        "runtime_speedup_median": 5.0,
                    },
                    "hybrid_vs_lattice": {
                        "erle_gain_db_median": 1.0,
                    },
                },
                "best_by_median_erle": {"name": "hybrid"},
            }
        ],
    }
    rows = build_rows(payload)
    assert rows[0]["hybrid_vs_lattice_erle_gain_db_median"] == 1.0
    summary = robust_summary(rows)
    assert summary["lattice_beats_fir_worst_count"] == 1
    assert summary["hybrid_beats_fir_median_count"] == 1

    csv_path = tmp_path / "summary.csv"
    write_scenario_csv(payload, csv_path)
    assert "lattice_vs_fir_erle_gain_db_median" in csv_path.read_text(encoding="utf-8")


def test_missing_echo_multiseed_input_message():
    message = missing_input_message(Path("echo-multiseed-benchmark.json"))
    assert "Input echo multi-seed benchmark JSON not found" in message
    assert "echo_multiseed_benchmark.py" in message
