# Graphene_BP Reproduction Report

Run directory: `results/smoke_train_only_imputer`

## What Was Reproduced

This report summarizes the feature-level Bio-Z to blood-pressure reproduction.
It uses the upstream pre-extracted feature table and the modernized
subject-specific AdaBoost runner in `scripts/run_reproduction.py`.

The paper abstract reference is DBP `0.2 +/- 4.5 mmHg` and SBP
`0.2 +/- 5.8 mmHg`; those values are used only as a high-level benchmark
because the public repository exposes processed features and experiment code,
not every figure-generation step.

## Pooled Test Metrics

| training_select | training_name | target | n | cc | me | std | mae | rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | SingleTrain_WithBaseValsalva | DBP | 94 | 0.699 | 7.359 | 4.281 | 7.683 | 8.502 |
| 3 | SingleTrain_WithBaseValsalva | SBP | 94 | 0.754 | -6.923 | 5.548 | 7.309 | 8.853 |

## Paper-Reference Comparison

| training_select | training_name | target | me | std | paper_me | paper_std | delta_abs_me | delta_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | SingleTrain_WithBaseValsalva | DBP | 7.359 | 4.281 | 0.200 | 4.500 | 7.159 | -0.219 |
| 3 | SingleTrain_WithBaseValsalva | SBP | -6.923 | 5.548 | 0.200 | 5.800 | 7.123 | -0.252 |

## Interpretation Notes

- `me` is prediction minus reference pressure, so positive values mean the
  model overestimates blood pressure.
- `std` is the sample standard deviation of prediction error and is the closest
  summary to the paper's `mean +/- std` abstract phrasing.
- The faithful default runner uses global mean imputation before splitting,
  matching the upstream code path; run `--imputer train_only` to quantify the
  safer no-test-leakage variant.
