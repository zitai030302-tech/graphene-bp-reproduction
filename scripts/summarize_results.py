#!/usr/bin/env python3
"""Build compact metrics and a Markdown report from a reproduction run."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


PAPER_REFERENCE = {
    "DBP": {"me": 0.2, "std": 4.5},
    "SBP": {"me": 0.2, "std": 5.8},
}


def corrcoef_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def pooled_metrics(rows: pd.DataFrame) -> dict[str, float | int]:
    err = rows["err"].to_numpy(dtype=float)
    y_true = rows["y_true"].to_numpy(dtype=float)
    y_pred = rows["y_pred"].to_numpy(dtype=float)
    return {
        "n": int(len(rows)),
        "cc": corrcoef_safe(y_true, y_pred),
        "me": float(np.mean(err)),
        "std": float(np.std(err, ddof=1)) if len(err) > 1 else float("nan"),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }


def build_pooled_table(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    keys = ["training_select", "training_name", "target", "split"]
    for group_key, rows in predictions.groupby(keys, dropna=False):
        record = dict(zip(keys, group_key))
        record.update(pooled_metrics(rows))
        records.append(record)
    return pd.DataFrame(records).sort_values(keys).reset_index(drop=True)


def build_paper_comparison(pooled: pd.DataFrame) -> pd.DataFrame:
    records = []
    test_rows = pooled[pooled["split"] == "test"]
    for _, row in test_rows.iterrows():
        ref = PAPER_REFERENCE.get(row["target"], {})
        records.append(
            {
                "training_select": row["training_select"],
                "training_name": row["training_name"],
                "target": row["target"],
                "n": row["n"],
                "me": row["me"],
                "std": row["std"],
                "mae": row["mae"],
                "rmse": row["rmse"],
                "cc": row["cc"],
                "paper_me": ref.get("me", math.nan),
                "paper_std": ref.get("std", math.nan),
                "delta_abs_me": abs(row["me"] - ref.get("me", math.nan)),
                "delta_std": row["std"] - ref.get("std", math.nan),
            }
        )
    return pd.DataFrame(records)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    shown = df.loc[:, columns].head(max_rows).copy()
    if shown.empty:
        return "_No rows._"
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def write_report(run_dir: Path, pooled: pd.DataFrame, comparison: pd.DataFrame) -> None:
    test_pooled = pooled[pooled["split"] == "test"].copy()
    report = f"""# Graphene_BP Reproduction Report

Run directory: `{run_dir}`

## What Was Reproduced

This report summarizes the feature-level Bio-Z to blood-pressure reproduction.
It uses the upstream pre-extracted feature table and the modernized
subject-specific AdaBoost runner in `scripts/run_reproduction.py`.

The paper abstract reference is DBP `0.2 +/- 4.5 mmHg` and SBP
`0.2 +/- 5.8 mmHg`; those values are used only as a high-level benchmark
because the public repository exposes processed features and experiment code,
not every figure-generation step.

## Pooled Test Metrics

{markdown_table(test_pooled, ["training_select", "training_name", "target", "n", "cc", "me", "std", "mae", "rmse"])}

## Paper-Reference Comparison

{markdown_table(comparison, ["training_select", "training_name", "target", "me", "std", "paper_me", "paper_std", "delta_abs_me", "delta_std"])}

## Interpretation Notes

- `me` is prediction minus reference pressure, so positive values mean the
  model overestimates blood pressure.
- `std` is the sample standard deviation of prediction error and is the closest
  summary to the paper's `mean +/- std` abstract phrasing.
- The faithful default runner uses global mean imputation before splitting,
  matching the upstream code path; run `--imputer train_only` to quantify the
  safer no-test-leakage variant.
"""
    (run_dir / "analysis_report.md").write_text(report, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)

    predictions_path = args.run_dir / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}")

    predictions = pd.read_csv(predictions_path)
    pooled = build_pooled_table(predictions)
    comparison = build_paper_comparison(pooled)
    pooled.to_csv(args.run_dir / "pooled_metrics.csv", index=False)
    comparison.to_csv(args.run_dir / "paper_comparison.csv", index=False)
    write_report(args.run_dir, pooled, comparison)
    print(f"wrote summaries in: {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

