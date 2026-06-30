#!/usr/bin/env python3
"""Create simple visual figures for non-specialist reporting."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import pandas as pd


RESULT_DIR = ROOT / "results" / "smoke"
FIG_DIR = ROOT / "results" / "figures"


def load_test_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(RESULT_DIR / "predictions.csv")
    summary = pd.read_csv(RESULT_DIR / "summary_by_subject.csv")
    return predictions[predictions["split"] == "test"].copy(), summary[summary["split"] == "test"].copy()


def save_prediction_trace(test_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    colors = {"DBP": "#2563eb", "SBP": "#dc2626"}
    for ax, target in zip(axes, ["DBP", "SBP"]):
        rows = test_df[test_df["target"] == target].reset_index(drop=True)
        x = range(1, len(rows) + 1)
        ax.plot(x, rows["y_true"], label="Reference", color="black", linewidth=2)
        ax.plot(x, rows["y_pred"], label="Predicted", color=colors[target], linewidth=2, alpha=0.85)
        ax.set_ylabel(f"{target} mmHg")
        ax.set_title(f"{target}: prediction follows the test segment trend")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Test sample order")
    fig.suptitle("Reference vs Predicted Blood Pressure", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "advisor_prediction_trace.png", dpi=200)
    plt.close(fig)


def save_scatter(test_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    colors = {"DBP": "#2563eb", "SBP": "#dc2626"}
    for ax, target in zip(axes, ["DBP", "SBP"]):
        rows = test_df[test_df["target"] == target]
        stats = summary_df[summary_df["target"] == target].iloc[0]
        ax.scatter(rows["y_true"], rows["y_pred"], s=36, color=colors[target], alpha=0.72)
        low = min(rows["y_true"].min(), rows["y_pred"].min()) - 2
        high = max(rows["y_true"].max(), rows["y_pred"].max()) + 2
        ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1.5, label="Perfect")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel("Reference mmHg")
        ax.set_ylabel("Predicted mmHg")
        ax.set_title(f"{target}: avg error {stats['me']:.1f}, typical miss {stats['mae']:.1f}")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    fig.suptitle("Predicted vs Reference", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "advisor_prediction_scatter.png", dpi=200)
    plt.close(fig)


def save_error_bars(summary_df: pd.DataFrame) -> None:
    summary_df = summary_df.set_index("target").loc[["DBP", "SBP"]]
    labels = ["DBP\nlow pressure", "SBP\nhigh pressure"]
    x = range(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar([i - width / 2 for i in x], summary_df["me"], width=width, label="Average error", color="#64748b")
    ax.bar([i + width / 2 for i in x], summary_df["mae"], width=width, label="Typical miss", color="#f59e0b")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("mmHg")
    ax.set_title("How far off was the model on the held-out test segment?")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    for i, target in enumerate(["DBP", "SBP"]):
        ax.text(i - width / 2, summary_df.loc[target, "me"], f"{summary_df.loc[target, 'me']:.1f}", ha="center", va="bottom" if summary_df.loc[target, "me"] >= 0 else "top")
        ax.text(i + width / 2, summary_df.loc[target, "mae"], f"{summary_df.loc[target, 'mae']:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "advisor_error_summary.png", dpi=200)
    plt.close(fig)


def save_data_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off")

    boxes = [
        (0.05, 0.62, "24-channel\nimpedance waves"),
        (0.05, 0.18, "Reference BP\nbeat-by-beat"),
        (0.30, 0.40, "Time align\nand clean"),
        (0.53, 0.40, "Extract heartbeat\nshape numbers"),
        (0.76, 0.40, "Personal\ncalibration"),
        (0.76, 0.08, "Estimated\nDBP / SBP"),
    ]
    for x, y, text in boxes:
        rect = plt.Rectangle((x, y), 0.17, 0.22, facecolor="#eff6ff", edgecolor="#2563eb", linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x + 0.085, y + 0.11, text, ha="center", va="center", fontsize=12)

    arrows = [
        ((0.22, 0.73), (0.30, 0.51)),
        ((0.22, 0.29), (0.30, 0.51)),
        ((0.47, 0.51), (0.53, 0.51)),
        ((0.70, 0.51), (0.76, 0.51)),
        ((0.845, 0.40), (0.845, 0.30)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", linewidth=2, color="#334155"))

    ax.text(
        0.5,
        0.94,
        "What data this type of BP study needs",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.02,
        "Key idea: impedance waves alone are not enough; they must be synchronized with trustworthy reference BP.",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#475569",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "advisor_data_pipeline.png", dpi=200)
    plt.close(fig)


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    test_df, summary_df = load_test_data()
    save_prediction_trace(test_df)
    save_scatter(test_df, summary_df)
    save_error_bars(summary_df)
    save_data_pipeline()
    print(f"wrote figures to: {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
