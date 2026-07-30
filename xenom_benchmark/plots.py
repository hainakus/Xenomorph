"""Matplotlib plotting helpers for benchmark reports."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


@dataclass
class PlotPaths:
    """Paths to the generated PNG plot files."""

    accuracy: Optional[Path] = None
    loss: Optional[Path] = None
    perplexity: Optional[Path] = None
    confidence: Optional[Path] = None
    confusion_matrix: Optional[Path] = None
    prediction_distribution: Optional[Path] = None
    training_history: Optional[Path] = None


def _save(fig: matplotlib.figure.Figure, path: Path) -> Optional[Path]:
    """Save a figure to ``path`` and close it, returning the path on success."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] failed to write plot {path}: {exc}")
        plt.close(fig)
        return None


def plot_confusion_matrix(
    cm: Dict[str, Dict[str, int]],
    out_path: Path,
) -> Optional[Path]:
    """Render a heatmap of the confusion matrix."""
    labels = sorted(cm.keys())
    if not labels:
        # Token-level MLM does not produce a base-level confusion matrix.
        return None
    matrix = np.array([[cm[t][p] for p in labels] for t in labels], dtype=int)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, matrix[i, j], ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    return _save(fig, out_path)


def plot_prediction_distribution(
    pred_dist: Dict[str, int],
    true_dist: Optional[Dict[str, float]] = None,
    out_path: Optional[Path] = None,
) -> Optional[Path]:
    """Bar plot comparing predicted and true token/base distributions."""
    if not pred_dist:
        return None

    # For token-level BPE vocabularies, keep only the top 50 most frequent
    # predicted tokens so the bar chart stays readable.
    max_labels = 50
    labels = sorted(pred_dist.keys(), key=lambda k: pred_dist[k], reverse=True)[:max_labels]
    pred_counts = np.array([pred_dist.get(b, 0) for b in labels], dtype=float)
    pred_freqs = pred_counts / pred_counts.sum() if pred_counts.sum() > 0 else pred_counts

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.25), 4))
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, pred_freqs, width, label="Predicted", color="steelblue")

    if true_dist:
        true_freqs = np.array([true_dist.get(b, 0.0) for b in labels], dtype=float)
        ax.bar(x + width / 2, true_freqs, width, label="True", color="coral")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Frequency")
    ax.set_title("Prediction Distribution")
    ax.legend()
    return _save(fig, out_path)


def plot_per_base_metrics(
    per_base: Dict[str, Dict[str, float]],
    out_path: Path,
) -> Optional[Path]:
    """Grouped bar chart of per-base accuracy, precision, recall and f1."""
    labels = sorted(per_base.keys())
    accuracy = [per_base[b]["accuracy"] for b in labels]
    precision = [per_base[b]["precision"] for b in labels]
    recall = [per_base[b]["recall"] for b in labels]
    f1 = [per_base[b]["f1"] for b in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    width = 0.2
    ax.bar(x - 1.5 * width, accuracy, width, label="Accuracy")
    ax.bar(x - 0.5 * width, precision, width, label="Precision")
    ax.bar(x + 0.5 * width, recall, width, label="Recall")
    ax.bar(x + 1.5 * width, f1, width, label="F1")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 1])
    ax.set_ylabel("Score")
    ax.set_title("Per-Base Metrics")
    ax.legend()
    return _save(fig, out_path)


def plot_loss(
    metrics: Dict[str, Any],
    out_path: Path,
) -> Optional[Path]:
    """Plot a simple cross-entropy / NLL summary (single value or history)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ce = metrics.get("cross_entropy")
    nll = metrics.get("nll")
    ce_lb = metrics.get("cross_entropy_lower_bound")

    if ce is not None and nll is not None and ce == nll:
        ax.bar(["NLL"], [ce], color="steelblue")
    else:
        names = []
        values = []
        if ce is not None:
            names.append("CE")
            values.append(ce)
        if nll is not None:
            names.append("NLL")
            values.append(nll)
        if ce_lb is not None:
            names.append("CE (lower bound)")
            values.append(ce_lb)
        ax.bar(names, values, color="steelblue")

    ax.set_ylabel("Nats")
    ax.set_title("Loss Summary")
    return _save(fig, out_path)


def _to_float(value: Any) -> Optional[float]:
    """Convert a JSON-safe value to float, treating the string 'inf' as infinity."""
    if value is None:
        return None
    if isinstance(value, float):
        return value
    if isinstance(value, (int, np.floating, np.integer)):
        return float(value)
    if value == "inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def plot_perplexity(
    metrics: Dict[str, Any],
    baselines: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> Optional[Path]:
    """Bar plot of model perplexity against baseline perplexities."""
    names = ["Model"]
    values = [_to_float(metrics.get("perplexity"))]

    for name, baseline in baselines.items():
        b_ppl = _to_float(baseline.get("perplexity"))
        if b_ppl is not None:
            names.append(name.title())
            values.append(b_ppl)

    valid = [(n, v) for n, v in zip(names, values) if v is not None and np.isfinite(v)]
    if not valid:
        return None

    names, values = zip(*valid)
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["steelblue"] + ["coral"] * (len(names) - 1)
    ax.bar(names, values, color=colors)
    ax.set_ylabel("Perplexity")
    ax.set_title("Perplexity vs Baselines")
    return _save(fig, out_path)


def plot_confidence(
    confidence: float,
    baselines: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> Optional[Path]:
    """Bar plot of average confidence against baseline confidences."""
    names = ["Model"]
    values = [confidence]

    for name, baseline in baselines.items():
        bc = baseline.get("average_confidence")
        if bc is not None:
            names.append(name.title())
            values.append(bc)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, values, color="steelblue")
    ax.set_ylim([0, 1])
    ax.set_ylabel("Average Confidence")
    ax.set_title("Average Confidence vs Baselines")
    return _save(fig, out_path)


def plot_accuracy(
    metrics: Dict[str, Any],
    baselines: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> Optional[Path]:
    """Bar plot of top-1 accuracy against baseline accuracies."""
    names = ["Model"]
    values = [metrics.get("top1_accuracy")]

    for name, baseline in baselines.items():
        b_acc = baseline.get("accuracy")
        if b_acc is not None:
            names.append(name.title())
            values.append(b_acc)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, values, color="steelblue")
    ax.set_ylim([0, 1])
    ax.set_ylabel("Accuracy")
    ax.set_title("Top-1 Accuracy vs Baselines")
    return _save(fig, out_path)


def plot_training_history(
    history: Sequence[Dict[str, Any]],
    out_path: Path,
) -> Optional[Path]:
    """Plot learning curves from a benchmark history file."""
    if not history:
        return None

    blocks = [h.get("block", i) for i, h in enumerate(history)]
    top1 = [h["metrics"].get("top1_accuracy", 0.0) for h in history]
    ppl = [_to_float(h["metrics"].get("perplexity")) for h in history]
    ce = [_to_float(h["metrics"].get("cross_entropy")) for h in history]
    conf = [h["metrics"].get("average_confidence", 0.0) for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0, 0]
    ax.plot(blocks, top1, marker="o", color="steelblue")
    ax.set_xlabel("Block")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("Accuracy over Blocks")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    valid_ppl = [(b, v) for b, v in zip(blocks, ppl) if v is not None and np.isfinite(v)]
    if valid_ppl:
        bx, vy = zip(*valid_ppl)
        ax.plot(bx, vy, marker="o", color="coral")
    ax.set_xlabel("Block")
    ax.set_ylabel("Perplexity")
    ax.set_title("Perplexity over Blocks")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    valid_ce = [(b, v) for b, v in zip(blocks, ce) if v is not None and np.isfinite(v)]
    if valid_ce:
        bx, vy = zip(*valid_ce)
        ax.plot(bx, vy, marker="o", color="green")
    ax.set_xlabel("Block")
    ax.set_ylabel("Cross-Entropy")
    ax.set_title("Cross-Entropy over Blocks")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(blocks, conf, marker="o", color="purple")
    ax.set_xlabel("Block")
    ax.set_ylabel("Average Confidence")
    ax.set_title("Confidence over Blocks")
    ax.grid(True, alpha=0.3)

    return _save(fig, out_path)


def generate_plots(
    metrics: Dict[str, Any],
    output_dir: Path,
) -> PlotPaths:
    """Generate all required PNG plots for a benchmark run."""
    paths = PlotPaths()

    paths.confusion_matrix = plot_confusion_matrix(
        metrics["confusion_matrix"],
        output_dir / "confusion_matrix.png",
    )
    paths.prediction_distribution = plot_prediction_distribution(
        metrics["prediction_distribution"],
        metrics.get("true_distribution"),
        output_dir / "prediction_distribution.png",
    )
    paths.loss = plot_loss(metrics, output_dir / "loss.png")
    paths.perplexity = plot_perplexity(
        metrics,
        metrics.get("baselines", {}),
        output_dir / "perplexity.png",
    )
    paths.confidence = plot_confidence(
        metrics.get("average_confidence", 0.0),
        metrics.get("baselines", {}),
        output_dir / "confidence.png",
    )
    paths.accuracy = plot_accuracy(
        metrics,
        metrics.get("baselines", {}),
        output_dir / "accuracy.png",
    )

    return paths


def generate_history_plot(
    history: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> Optional[Path]:
    """Generate the training-history learning-curve plot."""
    return plot_training_history(history, output_dir / "training_history.png")
