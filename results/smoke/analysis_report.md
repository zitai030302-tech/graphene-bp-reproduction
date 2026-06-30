# Graphene_BP Reproduction Report

Run directory: `results/smoke`

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
| 3 | SingleTrain_WithBaseValsalva | DBP | 94 | 0.704 | 3.667 | 4.522 | 4.825 | 5.803 |
| 3 | SingleTrain_WithBaseValsalva | SBP | 94 | 0.708 | -7.285 | 5.910 | 8.021 | 9.361 |

## Paper-Reference Comparison

| training_select | training_name | target | me | std | paper_me | paper_std | delta_abs_me | delta_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | SingleTrain_WithBaseValsalva | DBP | 3.667 | 4.522 | 0.200 | 4.500 | 3.467 | 0.022 |
| 3 | SingleTrain_WithBaseValsalva | SBP | -7.285 | 5.910 | 0.200 | 5.800 | 7.485 | 0.110 |

## Interpretation Notes

- `me` is prediction minus reference pressure, so positive values mean the
  model overestimates blood pressure.
- `std` is the sample standard deviation of prediction error and is the closest
  summary to the paper's `mean +/- std` abstract phrasing.
- The faithful default runner uses global mean imputation before splitting,
  matching the upstream code path; run `--imputer train_only` to quantify the
  safer no-test-leakage variant.
