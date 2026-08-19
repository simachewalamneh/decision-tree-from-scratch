# Hyperparameter Search: Grid Size Comparison

We tested two hyperparameter search grids via 5-fold cross-validation on
the Wine training set (143 samples), to check whether a larger, more
exhaustive search would find a better configuration than a smaller one.

## Grids tested

**Grid 1 (small):** 36 combinations
```python
{
    "criterion": ["gini", "entropy"],
    "max_depth": [1, 2, 3, 4, 5, 6],
    "min_samples_leaf": [1, 3, 5],
}
```

**Grid 2 (expanded):** 270 combinations — adds `accuracy` as a third
splitting criterion, and sweeps `min_samples_split` (previously left
fixed at its default) across five values.
```python
{
    "criterion": ["gini", "entropy", "accuracy"],
    "max_depth": [1, 2, 3, 4, 5, 6],
    "min_samples_split": [2, 4, 6, 8, 10],
    "min_samples_leaf": [1, 3, 5],
}
```

## Results

| | Grid 1 (36 combos) | Grid 2 (270 combos) |
|---|---|---|
| Search time (5-fold CV) | **7.8s** | 61.0s |
| Best combination found | `criterion=entropy, max_depth=4, min_samples_leaf=1` | `criterion=entropy, max_depth=4, min_samples_split=2, min_samples_leaf=1` |
| Best CV accuracy | 0.923 | 0.923 |
| Final test accuracy | **0.914** | **0.914** |

## Conclusion

The expanded grid took **7.8x longer** to search (61.0s vs 7.8s) but
converged on the same effective configuration and identical final test
accuracy (0.914). The additional parameters it covered — `accuracy` as
a splitting criterion, and `min_samples_split` values from 2 to 10 —
did not change the outcome:

- `accuracy` never outperformed `gini`/`entropy` for any depth/leaf
  combination on this dataset.
- `min_samples_split=2` (the most permissive setting, effectively "no
  extra restriction beyond `min_samples_leaf`") always won, meaning
  this particular stopping condition was never the limiting factor for
  Wine's tree structure at these depths.

**We use Grid 1 (the smaller, 36-combination search) going forward.**
It reaches the same result nearly 8x faster, and the expanded grid's
extra coverage — while worth testing once, to confirm nothing was being
missed — added computation without added benefit for this dataset. This
is not a general claim that `accuracy` or `min_samples_split` are never
useful hyperparameters; it's specific to Wine's particular structure at
the depths tested here.
