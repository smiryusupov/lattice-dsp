# Tuning `reflection_update_period`

`AdaptiveLatticeLadderNLMS` can update ladder/numerator taps every sample while
updating the more expensive reflection/denominator parameters every `K` samples.
The helper `tune_reflection_update_period` runs a small validation sweep and
returns the fastest period that satisfies quality and stability constraints.

```python
result = lattice_dsp.tune_reflection_update_period(
    x,
    desired,
    periods=[1, 2, 4, 8, 16, 32],
    order=4,
    max_tail_mse_ratio=1.5,
    max_worst_tail_mse_ratio=2.0,
)

print(result["recommended_period"])
print(result["warnings"])
print(result["metadata"]["recommendation_scope"])
```

For 1-D inputs, the median and worst-case tail-MSE ratios are the same and the
recommendation is labelled `single_signal`. The result includes a warning to
make clear that the chosen period is validated only on that signal:

```python
assert result["metadata"]["recommendation_scope"] == "single_signal"
assert result["warnings"]
```

For 2-D inputs, each row is treated as an independent validation trial, so the
worst-case threshold is evaluated across rows and the recommendation can be
labelled `robust` once `n_trials >= min_trials_for_robust`:

```python
# X and D have shape (trials, samples)
result = lattice_dsp.tune_reflection_update_period(
    X,
    D,
    periods=[1, 2, 4, 8, 16, 32],
    order=4,
    max_tail_mse_ratio=1.5,
    max_worst_tail_mse_ratio=2.0,
)
```

The first candidate period is used as the quality baseline. In most experiments
that should be period `1`.

By default, the helper uses `scale_reflection_mu_by_period=True`, so period `K`
applies `mu_reflection * K` on update samples. This keeps the effective
reflection learning rate comparable across periods.


The default `min_trials_for_robust=2` is intentionally modest. For publishable
or deployment-facing benchmarks, prefer several independent rows/signals and
inspect both `tail_mse_ratio_median` and `tail_mse_ratio_worst`.
