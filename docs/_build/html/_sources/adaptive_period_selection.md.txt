# Choosing `reflection_update_period`

Adaptive lattice-ladder filtering has two different update costs:

1. **tap / ladder updates** are cheap and normally run every sample;
2. **reflection / denominator updates** are more expensive and can be noisier.

`reflection_update_period` decimates the reflection update while still updating
the taps every sample.  With `scale_reflection_mu_by_period=True`, period `K`
applies `K * mu_reflection` when the denominator is updated, preserving a similar
effective learning rate.

## Practical interpretation

A useful workflow is:

```bash
python benchmarks/adaptive_period_sweep.py \
  --periods 1 2 4 8 16 32 \
  --samples 20000 \
  --repeats 5 \
  --output adaptive-period-sweep-scaled.json \
  --csv-output adaptive-period-sweep-scaled.csv

python benchmarks/adaptive_period_report.py adaptive-period-sweep-scaled.json \
  --markdown-output adaptive-period-report.md \
  --csv-output adaptive-period-pareto.csv \
  --max-tail-mse-ratio 1.5
```

The report chooses the fastest period whose tail MSE remains within the chosen
quality threshold.  A threshold of `1.5` means "allow at most 50% worse tail MSE
than period 1".

## Early results

On a representative Linux laptop with a 4th-order reflection model and
20,000 samples, scaled periodic updates produced the following pattern:

- period `1`: best reference quality, no speedup;
- period `8`: roughly `9.6x` faster with tail MSE very close to period `1`;
- period `16` and `32`: even faster, still close in tail MSE for this synthetic
  problem;
- fixed, unscaled periodic updates under-trained the denominator and produced
  much worse tail MSE.

This suggests `reflection_update_period=8` is a good conservative default for
examples, while `16` or `32` may be attractive when the system is slowly varying.
Run the sweep on the intended data distribution and report the selected default with the resulting tail-MSE and coefficient-error evidence.

## Notes

The period-scaling strategy is not a universal optimizer.  It keeps the average
step size comparable, but it still samples denominator gradients less often.  In
nonstationary echo-cancellation or equalization problems, long periods can miss
fast changes.  For those cases, start with period `1` or `2`, then increase the
period only after convergence is stable.
