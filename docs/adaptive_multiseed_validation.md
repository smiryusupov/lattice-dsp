# Multi-seed adaptive-period validation

Single-seed sweeps are useful, but they can pick a period that happens to work
well for one random input sequence.  The multi-seed sweep runs the same adaptive
IIR identification problem across several random seeds and aggregates both
speed and quality.

Run:

```bash
python benchmarks/adaptive_multiseed_sweep.py \
  --seeds 100 101 102 103 104 \
  --periods 1 2 4 8 16 32 \
  --samples 20000 \
  --repeats 3 \
  --output adaptive-multiseed-sweep.json \
  --csv-output adaptive-multiseed-aggregate.csv \
  --per-seed-csv-output adaptive-multiseed-per-seed.csv
```

Then summarize it:

```bash
python benchmarks/adaptive_multiseed_report.py adaptive-multiseed-sweep.json \
  --markdown-output adaptive-multiseed-report.md \
  --csv-output adaptive-multiseed-pareto.csv \
  --max-tail-mse-ratio-median 1.5 \
  --max-tail-mse-ratio-worst 2.0
```

The single-seed report recommends the fastest period that satisfies a tail-MSE
ratio threshold for one realization.  The multi-seed report is stricter: it can
require both the median tail-MSE ratio and the worst-case tail-MSE ratio to stay
below user-specified limits.

Important aggregate metrics:

- `speedup_vs_period1_median`: median speedup across seeds relative to period 1.
- `speedup_vs_period1_min`: worst speedup across seeds.
- `tail_mse_ratio_median`: median tail-MSE ratio across seeds relative to period
  1 for that seed.
- `tail_mse_ratio_max`: worst tail-MSE ratio across seeds.
- `stability_margin_min`: minimum `1 - max(abs(reflection))` across seeds.

A good default period should be fast, have low median tail-MSE ratio, have an
acceptable worst-case tail-MSE ratio, and keep a healthy stability margin.
