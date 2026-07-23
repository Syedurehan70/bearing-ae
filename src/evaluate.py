"""Scoring, thresholding and figures.

The threshold is set from healthy validation windows at the *training* load only.
Fault labels are never used to pick it. That constraint is what makes the
reported false-positive rate under load shift meaningful.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402


def choose_threshold(healthy_val_scores: np.ndarray, quantile: float) -> float:
    return float(np.quantile(healthy_val_scores, quantile))


def detection_metrics(
    scores: np.ndarray, is_faulty: np.ndarray, threshold: float
) -> dict[str, float]:
    flagged = scores > threshold
    healthy = ~is_faulty.astype(bool)
    faulty = is_faulty.astype(bool)

    out: dict[str, float] = {
        "n_windows": int(scores.size),
        "n_healthy": int(healthy.sum()),
        "n_faulty": int(faulty.sum()),
        "false_positive_rate": float(flagged[healthy].mean()) if healthy.any() else float("nan"),
        "true_positive_rate": float(flagged[faulty].mean()) if faulty.any() else float("nan"),
    }
    if healthy.any() and faulty.any():
        out["roc_auc"] = float(roc_auc_score(is_faulty, scores))
        out["average_precision"] = float(average_precision_score(is_faulty, scores))
    return out


def per_group_recall(
    scores: np.ndarray, groups: np.ndarray, threshold: float
) -> dict[str, float]:
    return {
        str(g): float((scores[groups == g] > threshold).mean())
        for g in sorted(set(groups.tolist()))
    }


def plot_score_distributions(
    scores: np.ndarray, is_faulty: np.ndarray, threshold: float, path: Path, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    log_scores = np.log10(scores + 1e-12)
    bins = np.linspace(log_scores.min(), log_scores.max(), 60)
    ax.hist(log_scores[~is_faulty.astype(bool)], bins=bins, alpha=0.65, label="healthy")
    ax.hist(log_scores[is_faulty.astype(bool)], bins=bins, alpha=0.65, label="faulty")
    ax.axvline(np.log10(threshold), color="k", ls="--", lw=1.2, label="threshold")
    ax.set_xlabel("log10 reconstruction error")
    ax.set_ylabel("windows")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_fpr_by_load(fpr_by_load: dict[int, float], train_load: int, path: Path) -> None:
    loads = sorted(fpr_by_load)
    values = [fpr_by_load[load] for load in loads]
    colours = ["#3b6ea5" if load == train_load else "#c25b45" for load in loads]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(load) for load in loads], values, color=colours)
    ax.set_xlabel("motor load (hp)")
    ax.set_ylabel("false positive rate on healthy windows")
    ax.set_title("Healthy false alarms under operating-condition shift")
    ax.axhline(values[loads.index(train_load)], color="k", ls=":", lw=1)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_history(history: dict[str, list[float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train"], label="train")
    ax.plot(history["val"], label="val (healthy)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
