from benchmarks.adaptive_multiseed_report import (
    recommend_period,
    robust_pareto_rows,
    missing_input_message,
)
from benchmarks.adaptive_multiseed_sweep import aggregate_rows, parse_seeds, quantile


def test_parse_seeds_accepts_commas_and_spaces():
    assert parse_seeds(["100,101", "102", "101"]) == [100, 101, 102]


def test_quantile_interpolates():
    assert quantile([1.0, 3.0], 0.5) == 2.0
    assert quantile([1.0, 2.0, 10.0], 0.9) == 8.4


def test_aggregate_rows_tracks_worst_case_quality():
    rows = [
        {
            "seed": 1,
            "reflection_update_period": 8,
            "median_s": 0.1,
            "speedup_vs_period1": 5.0,
            "mse_tail": 0.01,
            "tail_mse_ratio_vs_period1": 1.1,
            "mse_total": 0.02,
            "reflection_l2_error": 0.1,
            "taps_l2_error": 0.2,
            "stability_margin": 0.8,
        },
        {
            "seed": 2,
            "reflection_update_period": 8,
            "median_s": 0.2,
            "speedup_vs_period1": 4.0,
            "mse_tail": 0.03,
            "tail_mse_ratio_vs_period1": 2.0,
            "mse_total": 0.04,
            "reflection_l2_error": 0.3,
            "taps_l2_error": 0.4,
            "stability_margin": 0.7,
        },
    ]
    aggregate = aggregate_rows(rows, [8])
    assert aggregate[0]["tail_mse_ratio_median"] == 1.55
    assert aggregate[0]["tail_mse_ratio_max"] == 2.0
    assert aggregate[0]["stability_margin_min"] == 0.7


def test_robust_recommendation_uses_worst_case_constraint():
    rows = [
        {
            "reflection_update_period": 8,
            "median_s_median": 0.2,
            "tail_mse_ratio_median": 1.1,
            "tail_mse_ratio_max": 1.4,
            "stability_margin_min": 0.5,
        },
        {
            "reflection_update_period": 32,
            "median_s_median": 0.1,
            "tail_mse_ratio_median": 1.2,
            "tail_mse_ratio_max": 3.0,
            "stability_margin_min": 0.5,
        },
    ]
    row = recommend_period(
        rows,
        max_tail_mse_ratio_median=1.5,
        max_tail_mse_ratio_worst=2.0,
        min_stability_margin=0.1,
        prefer="fastest",
    )
    assert row["reflection_update_period"] == 8


def test_robust_pareto_rows_uses_worst_tail_ratio():
    rows = [
        {"reflection_update_period": 1, "median_s_median": 1.0, "tail_mse_ratio_max": 1.0},
        {"reflection_update_period": 8, "median_s_median": 0.2, "tail_mse_ratio_max": 1.2},
        {
            "reflection_update_period": 16,
            "median_s_median": 0.3,
            "tail_mse_ratio_max": 1.3,
        },  # dominated
    ]
    assert [row["reflection_update_period"] for row in robust_pareto_rows(rows)] == [8, 1]


def test_missing_multiseed_input_message_suggests_sweep_command(tmp_path):
    path = tmp_path / "adaptive-multiseed-sweep.json"
    message = missing_input_message(path)
    assert "Input multi-seed sweep JSON not found" in message
    assert "adaptive_multiseed_sweep.py" in message
    assert "--output adaptive-multiseed-sweep.json" in message
