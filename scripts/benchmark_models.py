"""Compare simple regressors under explicit subject-generalization protocols."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


def synthetic(seed=42, subjects=6, samples=60):
    """Generated cohort with subject offsets; not patient or device data."""
    rng = np.random.default_rng(seed)
    rows = []
    for subject in range(subjects):
        baseline = rng.normal(0, 12)
        for sample in range(samples):
            values = rng.normal(size=4)
            target = 120 + baseline + values[0] * 7 - values[1] * 4 + rng.normal(0, 3)
            rows.append({"subject_id": f"SYN{subject:02d}", "f0": values[0], "f1": values[1],
                         "f2": values[2], "f3": values[3], "SBP": target})
    frame = pd.DataFrame(rows)
    frame.loc[rng.choice(len(frame), len(frame) // 20, replace=False), "f2"] = np.nan
    return frame


def splits(frame, protocol, group_column="subject_id", seed=42):
    if frame[group_column].isna().any() or frame[group_column].nunique() < 2:
        raise ValueError("at least two complete subject groups required")
    index = np.arange(len(frame))
    if protocol == "random":
        train, test = train_test_split(index, test_size=.25, random_state=seed)
        return [(0, train, test)]
    if protocol == "group_holdout":
        splitter = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed)
    elif protocol == "loso":
        splitter = LeaveOneGroupOut()
    else:
        raise ValueError("unknown evaluation protocol")
    return [(i, tr, te) for i, (tr, te) in enumerate(splitter.split(index, groups=frame[group_column]))]


def models(seed=42):
    return {
        "mean": make_pipeline(SimpleImputer(keep_empty_features=True), DummyRegressor()),
        "ridge": make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler(), Ridge(alpha=1)),
        "random_forest": make_pipeline(SimpleImputer(keep_empty_features=True), RandomForestRegressor(n_estimators=80, min_samples_leaf=3, random_state=seed, n_jobs=1)),
        "adaboost": make_pipeline(SimpleImputer(keep_empty_features=True), AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=3, random_state=seed), n_estimators=60, random_state=seed)),
    }


def run(frame, features, target, output, seed=42, group_column="subject_id", protocols=("random", "group_holdout", "loso")):
    if not features or len(set(features)) != len(features) or {target, group_column, "SBP", "DBP", "source_row"} & set(features):
        raise ValueError("features must be unique and cannot contain targets or identifiers")
    required = set(features) | {target, group_column}
    if not required <= set(frame.columns):
        raise ValueError(f"missing columns: {sorted(required - set(frame.columns))}")
    if frame[target].isna().any() or not np.isfinite(frame[target].to_numpy(dtype=float)).all():
        raise ValueError("target must be finite and complete; do not silently impute labels")
    x = frame[features].to_numpy(dtype=float)
    if np.isinf(x).any():
        raise ValueError("infinite feature values are not supported")
    y = frame[target].to_numpy(dtype=float)
    rows, predictions, membership = [], [], []
    for protocol in protocols:
        for fold, train, test in splits(frame, protocol, group_column, seed):
            train_groups = set(frame.iloc[train][group_column])
            test_groups = set(frame.iloc[test][group_column])
            overlap = sorted(map(str, train_groups & test_groups))
            if protocol != "random" and overlap:
                raise AssertionError("group leakage")
            for role, indices in (("train", train), ("test", test)):
                membership.extend({"protocol": protocol, "fold": fold, "source_row": int(i), "role": role, "group": str(frame.iloc[i][group_column])} for i in indices)
            for name, estimator in models(seed).items():
                estimator.fit(x[train], y[train])
                pred = estimator.predict(x[test])
                error = pred - y[test]
                rows.append({"protocol": protocol, "fold": fold, "model": name,
                             "n_train": len(train), "n_test": len(test),
                             "train_test_group_overlap": len(overlap),
                             "mae": float(mean_absolute_error(y[test], pred)),
                             "rmse": float(mean_squared_error(y[test], pred) ** .5),
                             "mean_error": float(error.mean())})
                predictions.extend({"protocol": protocol, "fold": fold, "model": name, "source_row": int(i),
                                    "group": str(frame.iloc[i][group_column]), "true": float(true), "predicted": float(p)}
                                   for i, true, p in zip(test, y[test], pred))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(output / "fold_metrics.csv", index=False)
    prediction_frame = pd.DataFrame(predictions)
    prediction_frame.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(membership).to_csv(output / "split_membership.csv", index=False)
    # Report pooled held-out predictions instead of averaging RMSE across folds.
    pooled = []
    for (protocol, name), group in prediction_frame.groupby(["protocol", "model"]):
        error = group.predicted - group.true
        pooled.append({"protocol": protocol, "model": name, "n_test": len(group),
                       "mae": float(error.abs().mean()), "rmse": float(np.sqrt((error ** 2).mean()))})
    summary = pd.DataFrame(pooled)
    summary.to_csv(output / "summary.csv", index=False)
    config = {"seed": seed, "features": features, "target": target, "group_column": group_column,
              "protocols": list(protocols), "preprocessing": "fold-local imputation; fold-local scaling for ridge",
              "python": platform.python_version(), "sklearn": sklearn.__version__,
              "numpy": np.__version__, "pandas": pd.__version__,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "table_sha256": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
              "hyperparameters": {name: repr(model) for name, model in models(seed).items()}}
    (output / "manifest.json").write_text(json.dumps(config, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--features", help="comma-separated numeric columns; never target or subject IDs")
    parser.add_argument("--target", default="SBP")
    parser.add_argument("--group-column", default="subject_id")
    parser.add_argument("--output", default="results/model_benchmark")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if bool(args.csv) == bool(args.synthetic):
        parser.error("choose exactly one of --csv or --synthetic")
    frame = synthetic(args.seed) if args.synthetic else pd.read_csv(args.csv)
    features = args.features.split(",") if args.features else ["f0", "f1", "f2", "f3"] if args.synthetic else None
    if features is None:
        parser.error("--features is required for a real CSV")
    summary = run(frame, features, args.target, args.output, args.seed, args.group_column)
    print(summary.to_string(index=False))
