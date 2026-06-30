# Reproduction Log

Date: 2026-05-21

## Environment

- Workspace: `/Users/lizitai/Desktop/四电极阻抗/graphene_bp_reproduction`
- Working Python used for verification: `/opt/miniconda3/bin/python3`
- Available package versions during verification:
  - `numpy 2.3.5`
  - `pandas 3.0.0`
  - `scikit-learn 1.7.2`
- Upstream original dependency pins are preserved in `upstream/requirements.txt`.

## Assets

- Downloaded upstream source files:
  - `upstream/BP_BioZ_v0.py`
  - `upstream/UtilityFuncs.py`
  - `upstream/config.hjson`
  - `upstream/requirements.txt`
- Downloaded feature table:
  - `upstream/Data/features/2020-11-05/f2_ma20_mn1_mean_all.csv`
  - Size: 36 MB
  - Line count: 29810 including header

## Smoke Test

Command:

```bash
/opt/miniconda3/bin/python3 scripts/run_reproduction.py --smoke
/opt/miniconda3/bin/python3 scripts/summarize_results.py results/smoke
```

Observed settings and output:

- Rows loaded: 29809
- Matched feature count: 54
- Downsample rate parsed from filename: 20
- Scenario: `training_select=3`, `SingleTrain_WithBaseValsalva`
- Subject: 1
- Valid test fold found: `postExerciseHgcp`, `n=94`
- Output directory: `results/smoke`

Pooled test metrics:

| Target | N | CC | ME | STD | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DBP | 94 | 0.704 | 3.667 | 4.522 | 4.825 | 5.803 |
| SBP | 94 | 0.708 | -7.285 | 5.910 | 8.021 | 9.361 |

## Train-Only Imputer Sensitivity

Command:

```bash
/opt/miniconda3/bin/python3 scripts/run_reproduction.py --smoke --imputer train_only --output-dir results/smoke_train_only_imputer
/opt/miniconda3/bin/python3 scripts/summarize_results.py results/smoke_train_only_imputer
```

Pooled test metrics:

| Target | N | CC | ME | STD | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DBP | 94 | 0.699 | 7.359 | 4.281 | 7.683 | 8.502 |
| SBP | 94 | 0.754 | -6.923 | 5.548 | 7.309 | 8.853 |

Interpretation: removing test-set information from mean imputation changed DBP bias
substantially in this smoke slice. This supports treating the upstream global
imputation path as a faithful reproduction mode, not as the final evaluation
policy for a new project.

## Fixed-Seed Reproducibility Check

Commands:

```bash
/opt/miniconda3/bin/python3 scripts/run_reproduction.py --training-select 0 --subjects 1 --max-folds 1 --output-dir results/repro_shuffle_a
/opt/miniconda3/bin/python3 scripts/run_reproduction.py --training-select 0 --subjects 1 --max-folds 1 --output-dir results/repro_shuffle_b
cmp -s results/repro_shuffle_a/summary_by_subject.csv results/repro_shuffle_b/summary_by_subject.csv
cmp -s results/repro_shuffle_a/predictions.csv results/repro_shuffle_b/predictions.csv
```

Result: both `cmp` checks exited with code 0, so the fixed-seed shuffled fold
and AdaBoost output are byte-identical for this minimal check.

## Full-Matrix Readiness

The full matrix command is implemented in `README.md` but was not launched
during this verification pass because it expands to many nested grid searches.
The largest CV scenario is `training_select=1`, where each of 6 subjects has
10 outer folds and each fold trains separate DBP/SBP AdaBoost models with
inner CV. The runner is ready for it; use:

```bash
/opt/miniconda3/bin/python3 scripts/run_reproduction.py --training-select all --subjects 1,2,3,4,5,6 --output-dir results/full
/opt/miniconda3/bin/python3 scripts/summarize_results.py results/full
```
