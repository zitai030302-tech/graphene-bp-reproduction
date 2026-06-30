#!/usr/bin/env python3
"""Feature-level Graphene_BP reproduction runner.

The implementation mirrors the upstream experiment structure while staying
compatible with modern pandas and scikit-learn versions.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_FILE = (
    ROOT / "upstream" / "Data" / "features" / "2020-11-05" / "f2_ma20_mn1_mean_all.csv"
)


TRAINING_NAME_LIST = [
    "ShuffleCV_hgcp",
    "NoShuffleCV_hgcp",
    "SingleTrain_hgcp",
    "SingleTrain_WithBaseValsalva",
    "SingleTrain_S1All_TestNextDay",
    "ShuffleCV_valsalva",
    "NoShuffleCV_valsalva",
]

CV_LIST = [1, 1, 0, 0, 0, 1, 1]
SHUFFLE_LIST = [1, 0, 0, 0, 0, 1, 0]

SETUPS_NAME_MATCH_STR = [
    "subject1_day1_.*hgcp|subject2_day2_.*hgcp|subject3_day2_.*hgcp|subject4_day3_.*hgcp|subject5_day3_.*hgcp|subject6_day4_.*hgcp",
    "subject1_day1_.*hgcp|subject2_day2_.*hgcp|subject3_day2_.*hgcp|subject4_day3_.*hgcp|subject5_day3_.*hgcp|subject6_day4_.*hgcp",
    ".*",
    "subject1_day1|subject2_day2|subject3_day2|subject4_day3|subject5_day3|subject6_day4",
    "subject1",
    "valsalva",
    "valsalva",
]

TRAINING_MATCH_STR = [
    "",
    "",
    "subject1_day1_.*hgcp|subject2_day2_.*hgcp|subject3_day2_.*hgcp|subject4_day3_.*hgcp|subject5_day3_.*hgcp|subject6_day4_.*hgcp",
    "hgcp|baseline|valsalva",
    "subject1_day1",
    "",
    "",
]

TESTING_MATCH_STR = [
    [],
    [],
    ["postExerciseHgcp", "postMealHgcp", "postSweatBaseline"],
    ["postExerciseHgcp", "postMealHgcp", "postSweatBaseline"],
    ["subject1_day4.*setup01_hgcp", "subject1_day4.*setup02_hgcp", "subject1_day4.*valsalva"],
    [],
    [],
]

FEATURE_PATTERNS = [
    r"^BM[1234](_BM[1234])*_[A-Z2]*__((?!T$))",
    r"^BM[1234](_BM[1234])*_[A-Z2]*__((?!IBI)(?!T$))",
]

INFO_COLUMNS = [
    "subject_id",
    "setup_n",
    "trial_n",
    "P_MS__T",
    "Sample_c",
    "exp_setup_name",
]


@dataclass
class RunOptions:
    feature_file: str
    output_dir: str
    training_select: list[int]
    subjects: list[int]
    feature_select: int
    seed: int
    n_fold: int
    row_remove_nan_perc: float
    imputer: str
    n_jobs: int
    max_folds: int | None
    timestamp: str


def parse_int_list(value: str, *, allow_all: bool, max_value: int | None = None) -> list[int]:
    value = value.strip()
    if allow_all and value == "all":
        if max_value is None:
            raise ValueError("max_value is required for all")
        return list(range(max_value + 1))
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_downsample_rate(feature_file: Path) -> int:
    match = re.search(r"f\d+_ma(\d+)_", feature_file.name)
    return int(match.group(1)) if match else 20


def contains_regex(values: pd.Series, pattern: str) -> np.ndarray:
    return values.astype(str).str.contains(pattern, regex=True, na=False).to_numpy()


def select_features(data: pd.DataFrame, feature_select: int) -> list[str]:
    pattern = re.compile(FEATURE_PATTERNS[feature_select])
    features = [column for column in data.columns if pattern.match(column)]
    if not features:
        raise ValueError(f"No features matched feature_select={feature_select}")
    return features


def clean_subject_table(
    all_data: pd.DataFrame,
    subject_id: int,
    training_select: int,
    downsample_rate: int,
    row_remove_nan_perc: float,
) -> pd.DataFrame:
    subject_data = all_data[all_data["subject_id"] == subject_id].copy()
    setup_mask = contains_regex(subject_data["exp_setup_name"], SETUPS_NAME_MATCH_STR[training_select])
    filtered = subject_data.loc[setup_mask].copy()
    if SHUFFLE_LIST[training_select] == 1:
        step = max(int(downsample_rate / 2), 1)
        filtered = filtered.iloc[::step].copy()
    nan_limit = row_remove_nan_perc * filtered.shape[1]
    cleaned = filtered[pd.isnull(filtered).sum(axis=1) <= nan_limit].copy()
    return cleaned.reset_index(drop=False).rename(columns={"index": "source_row"})


def build_splits(data: pd.DataFrame, training_select: int, n_fold: int, seed: int) -> list[tuple[int, np.ndarray, np.ndarray, str]]:
    if CV_LIST[training_select] == 1:
        n_splits = min(n_fold, len(data))
        if n_splits < 2:
            return []
        kf = KFold(
            n_splits=n_splits,
            shuffle=bool(SHUFFLE_LIST[training_select]),
            random_state=seed if SHUFFLE_LIST[training_select] else None,
        )
        return [
            (fold_id, train_idx, test_idx, f"kfold_{fold_id:02d}")
            for fold_id, (train_idx, test_idx) in enumerate(kf.split(data), start=1)
        ]

    splits = []
    train_pattern = TRAINING_MATCH_STR[training_select]
    exp_names = data["exp_setup_name"]
    train_idx = np.flatnonzero(contains_regex(exp_names, train_pattern))
    for fold_id, test_pattern in enumerate(TESTING_MATCH_STR[training_select], start=1):
        test_idx = np.flatnonzero(contains_regex(exp_names, test_pattern))
        if len(train_idx) and len(test_idx):
            splits.append((fold_id, train_idx, test_idx, test_pattern))
    return splits


def dynamic_ada_grid(n_train: int, n_features: int) -> dict[str, list[int]]:
    train_scale = max(n_train / 10, 1.0)
    feature_scale = max(n_features / 10, 1.0)
    estimator_power = max(int(math.log(train_scale, 2)) - 3, 3)
    depth_power = int(math.log(feature_scale, 2)) + 2
    return {
        "n_estimators": [x * pow(2, estimator_power) for x in [1, 2]],
        "estimator__max_depth": [x * pow(2, depth_power) for x in [1, 2]],
    }


def fit_adaboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    n_fold: int,
    n_jobs: int,
) -> GridSearchCV:
    base_tree = DecisionTreeRegressor(random_state=seed)
    model = AdaBoostRegressor(estimator=base_tree, random_state=seed)
    grid = dynamic_ada_grid(len(x_train), x_train.shape[1])
    inner_splits = min(max(n_fold - 1, 2), len(x_train))
    inner_cv = KFold(n_splits=inner_splits, shuffle=False)
    search = GridSearchCV(
        estimator=model,
        param_grid=grid,
        cv=inner_cv,
        scoring="neg_mean_squared_error",
        n_jobs=n_jobs,
    )
    search.fit(x_train, y_train)
    return search


def corrcoef_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def metric_row(
    rows: pd.DataFrame,
    *,
    training_select: int,
    subject_id: int,
    target: str,
    split: str,
    imputer: str,
) -> dict[str, object]:
    err = rows["err"].to_numpy(dtype=float)
    y_true = rows["y_true"].to_numpy(dtype=float)
    y_pred = rows["y_pred"].to_numpy(dtype=float)
    return {
        "training_select": training_select,
        "training_name": TRAINING_NAME_LIST[training_select],
        "subject_id": subject_id,
        "target": target,
        "split": split,
        "imputer": imputer,
        "n": int(len(rows)),
        "cc": corrcoef_safe(y_true, y_pred),
        "me": float(np.mean(err)) if len(err) else float("nan"),
        "std": float(np.std(err, ddof=1)) if len(err) > 1 else float("nan"),
        "mae": float(mean_absolute_error(y_true, y_pred)) if len(err) else float("nan"),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5) if len(err) else float("nan"),
    }


def prediction_rows(
    data: pd.DataFrame,
    indices: Iterable[int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    training_select: int,
    subject_id: int,
    target: str,
    split: str,
    fold_id: int,
    fold_label: str,
    best_params: dict[str, object],
) -> list[dict[str, object]]:
    rows = []
    indexed = data.iloc[list(indices)].reset_index(drop=True)
    for local_i, (_, info) in enumerate(indexed.iterrows()):
        pred = float(y_pred[local_i])
        truth = float(y_true[local_i])
        rows.append(
            {
                "training_select": training_select,
                "training_name": TRAINING_NAME_LIST[training_select],
                "subject_id": subject_id,
                "target": target,
                "split": split,
                "fold": fold_id,
                "fold_label": fold_label,
                "source_row": int(info["source_row"]),
                "sample": int(info["Sample"]) if "Sample" in info and not pd.isna(info["Sample"]) else None,
                "sample_c": info.get("Sample_c", None),
                "setup_n": info.get("setup_n", None),
                "trial_n": info.get("trial_n", None),
                "exp_setup_name": info.get("exp_setup_name", None),
                "y_true": truth,
                "y_pred": pred,
                "err": pred - truth,
                "n_estimators": best_params.get("n_estimators"),
                "max_depth": best_params.get("estimator__max_depth"),
            }
        )
    return rows


def run_subject(
    all_data: pd.DataFrame,
    *,
    subject_id: int,
    training_select: int,
    options: RunOptions,
    feature_names: list[str],
    downsample_rate: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    data = clean_subject_table(
        all_data,
        subject_id,
        training_select,
        downsample_rate,
        options.row_remove_nan_perc,
    )
    if len(data) < 24:
        print(f"skip subject={subject_id} training={training_select}: not enough rows ({len(data)})")
        return [], []

    splits = build_splits(data, training_select, options.n_fold, options.seed)
    if options.max_folds is not None:
        splits = splits[: options.max_folds]
    if not splits:
        print(f"skip subject={subject_id} training={training_select}: no valid splits")
        return [], []

    x_raw = data[feature_names]
    if options.imputer == "global":
        global_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        x_global = global_imputer.fit_transform(x_raw)
    else:
        x_global = None

    prediction_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []

    for target in ["DBP", "SBP"]:
        cached_model: GridSearchCV | None = None
        target_split_records: list[dict[str, object]] = []
        for fold_id, train_idx, test_idx, fold_label in splits:
            y_all = data[target].to_numpy(dtype=float)
            y_train = y_all[train_idx]
            y_test = y_all[test_idx]

            if options.imputer == "global":
                x_train = x_global[train_idx]
                x_test = x_global[test_idx]
            else:
                imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
                x_train = imputer.fit_transform(x_raw.iloc[train_idx])
                x_test = imputer.transform(x_raw.iloc[test_idx])

            if CV_LIST[training_select] == 0 and cached_model is not None:
                model = cached_model
            else:
                model = fit_adaboost(x_train, y_train, options.seed, options.n_fold, options.n_jobs)
                if CV_LIST[training_select] == 0:
                    cached_model = model

            best_params = dict(model.best_params_)
            train_pred = model.predict(x_train)
            test_pred = model.predict(x_test)

            train_records = prediction_rows(
                data,
                train_idx,
                y_train,
                train_pred,
                training_select=training_select,
                subject_id=subject_id,
                target=target,
                split="train",
                fold_id=fold_id,
                fold_label=fold_label,
                best_params=best_params,
            )
            test_records = prediction_rows(
                data,
                test_idx,
                y_test,
                test_pred,
                training_select=training_select,
                subject_id=subject_id,
                target=target,
                split="test",
                fold_id=fold_id,
                fold_label=fold_label,
                best_params=best_params,
            )
            target_split_records.extend(train_records)
            target_split_records.extend(test_records)

            print(
                "done",
                f"training={training_select}:{TRAINING_NAME_LIST[training_select]}",
                f"subject={subject_id}",
                f"target={target}",
                f"fold={fold_id}",
                f"test_n={len(test_idx)}",
                f"params={best_params}",
            )

        prediction_records.extend(target_split_records)
        target_df = pd.DataFrame(target_split_records)
        for split in ["train", "test"]:
            split_df = target_df[target_df["split"] == split]
            if not split_df.empty:
                summary_records.append(
                    metric_row(
                        split_df,
                        training_select=training_select,
                        subject_id=subject_id,
                        target=target,
                        split=split,
                        imputer=options.imputer,
                    )
                )

    return prediction_records, summary_records


def summarize_overall(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    grouped = []
    for keys, rows in summary.groupby(["training_select", "training_name", "target", "split", "imputer"], dropna=False):
        training_select, training_name, target, split, imputer = keys
        grouped.append(
            {
                "training_select": training_select,
                "training_name": training_name,
                "target": target,
                "split": split,
                "imputer": imputer,
                "subjects": ",".join(str(x) for x in sorted(rows["subject_id"].unique())),
                "n_subjects": int(rows["subject_id"].nunique()),
                "mean_cc": float(rows["cc"].mean()),
                "mean_me": float(rows["me"].mean()),
                "mean_std": float(rows["std"].mean()),
                "mean_mae": float(rows["mae"].mean()),
                "mean_rmse": float(rows["rmse"].mean()),
            }
        )
    return pd.DataFrame(grouped)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-file", type=Path, default=DEFAULT_FEATURE_FILE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--training-select", default="3")
    parser.add_argument("--subjects", default="1,2,3,4,5,6")
    parser.add_argument("--feature-select", type=int, default=0, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--n-fold", type=int, default=10)
    parser.add_argument("--row-remove-nan-perc", type=float, default=0.1)
    parser.add_argument("--imputer", choices=["global", "train_only"], default="global")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.smoke:
        args.training_select = "3"
        args.subjects = "1"
        if args.output_dir is None:
            args.output_dir = ROOT / "results" / "smoke"
    elif args.output_dir is None:
        args.output_dir = ROOT / "results" / "run"

    if not args.feature_file.exists():
        raise FileNotFoundError(
            f"Missing feature file: {args.feature_file}. Run scripts/download_assets.py first."
        )

    training_select = parse_int_list(args.training_select, allow_all=True, max_value=len(TRAINING_NAME_LIST) - 1)
    subjects = parse_int_list(args.subjects, allow_all=False)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    options = RunOptions(
        feature_file=str(args.feature_file),
        output_dir=str(args.output_dir),
        training_select=training_select,
        subjects=subjects,
        feature_select=args.feature_select,
        seed=args.seed,
        n_fold=args.n_fold,
        row_remove_nan_perc=args.row_remove_nan_perc,
        imputer=args.imputer,
        n_jobs=args.n_jobs,
        max_folds=args.max_folds,
        timestamp=timestamp,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"load features: {args.feature_file}")
    all_data = pd.read_csv(args.feature_file, quotechar='"', skipinitialspace=True)
    feature_names = select_features(all_data, args.feature_select)
    downsample_rate = parse_downsample_rate(args.feature_file)
    print(f"rows={len(all_data)} features={len(feature_names)} downsample_rate={downsample_rate}")

    all_predictions: list[dict[str, object]] = []
    all_summary: list[dict[str, object]] = []
    for train_sel in training_select:
        for subject_id in subjects:
            predictions, summary = run_subject(
                all_data,
                subject_id=subject_id,
                training_select=train_sel,
                options=options,
                feature_names=feature_names,
                downsample_rate=downsample_rate,
            )
            all_predictions.extend(predictions)
            all_summary.extend(summary)

    predictions_df = pd.DataFrame(all_predictions)
    summary_df = pd.DataFrame(all_summary)
    overall_df = summarize_overall(summary_df)

    predictions_df.to_csv(args.output_dir / "predictions.csv", index=False)
    summary_df.to_csv(args.output_dir / "summary_by_subject.csv", index=False)
    overall_df.to_csv(args.output_dir / "summary_overall.csv", index=False)
    (args.output_dir / "feature_names.txt").write_text("\n".join(feature_names) + "\n", encoding="utf-8")
    (args.output_dir / "run_config.json").write_text(
        json.dumps(asdict(options), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
