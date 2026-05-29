from benchmarks.adaptive_period_report import missing_input_message, pareto_rows, recommend_period


def test_recommend_period_prefers_fastest_within_quality_constraint():
    rows = [
        {
            "reflection_update_period": 1,
            "median_s": 1.0,
            "mse_tail": 1.0,
            "tail_mse_ratio_vs_first_period": 1.0,
            "stability_margin": 0.5,
        },
        {
            "reflection_update_period": 8,
            "median_s": 0.2,
            "mse_tail": 1.2,
            "tail_mse_ratio_vs_first_period": 1.2,
            "stability_margin": 0.5,
        },
        {
            "reflection_update_period": 32,
            "median_s": 0.1,
            "mse_tail": 3.0,
            "tail_mse_ratio_vs_first_period": 3.0,
            "stability_margin": 0.5,
        },
    ]
    row = recommend_period(rows, max_tail_mse_ratio=1.5, min_stability_margin=0.1, prefer="fastest")
    assert row["reflection_update_period"] == 8


def test_pareto_rows_removes_dominated_points():
    rows = [
        {"reflection_update_period": 1, "median_s": 1.0, "mse_tail": 0.1},
        {"reflection_update_period": 2, "median_s": 0.8, "mse_tail": 0.2},
        {"reflection_update_period": 4, "median_s": 0.9, "mse_tail": 0.3},  # dominated by period 2
    ]
    periods = [row["reflection_update_period"] for row in pareto_rows(rows)]
    assert periods == [2, 1]


def test_missing_input_message_suggests_scaled_sweep_command(tmp_path):
    path = tmp_path / "adaptive-period-sweep-scaled.json"
    message = missing_input_message(path)
    assert "Input sweep JSON not found" in message
    assert "adaptive_period_sweep.py" in message
    assert "--output adaptive-period-sweep-scaled.json" in message
