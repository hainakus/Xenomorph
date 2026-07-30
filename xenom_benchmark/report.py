"""Report generation: JSON, CSV, HTML and benchmark history."""

import csv
import html
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .metrics import MetricsResult
from .plots import PlotPaths, generate_history_plot, generate_plots
from .utils import BENCHMARK_VERSION, ensure_dir, now_iso


def _json_friendly(obj: Any) -> Any:
    """Recursively convert numpy / dataclass / inf values for JSON."""
    import numpy as np

    if isinstance(obj, (list, tuple)):
        return [_json_friendly(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _json_friendly(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer, np.floating)):
        val = float(obj)
    else:
        val = obj

    if isinstance(val, float):
        if not (val == val):  # NaN
            return None
        if val == float("inf"):
            return "inf"
        if val == float("-inf"):
            return "-inf"
    return val


def _metrics_to_dict(metrics: MetricsResult) -> Dict[str, Any]:
    """Convert a MetricsResult to a JSON-safe dict."""
    raw = asdict(metrics)
    return _json_friendly(raw)


@dataclass
class BenchmarkRecord:
    """A single benchmark evaluation record."""

    benchmark_version: str = BENCHMARK_VERSION
    timestamp: str = field(default_factory=now_iso)
    block: int = 0
    model_id: str = ""
    model_version: str = ""
    model_hash: Optional[str] = None
    checkpoint: Optional[str] = None
    genome_source: str = ""
    genome_hash: Optional[str] = None
    seed: int = 42
    num_samples: int = 0
    seq_length: int = 0
    mask_prob: float = 0.0
    grpc_address: str = ""
    vocab_size: Optional[int] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    region_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return _json_friendly(asdict(self))


def load_history(path: Path) -> List[Dict[str, Any]]:
    """Load benchmark history from ``path`` or return an empty list."""
    if not path.exists():
        return []
    with open(path, "r") as f:
        return json.load(f)


def save_history(path: Path, history: Sequence[Dict[str, Any]]) -> None:
    """Save benchmark history to ``path`` as JSON."""
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(_json_friendly(list(history)), f, indent=2)


def append_history(path: Path, record: BenchmarkRecord) -> List[Dict[str, Any]]:
    """Append ``record`` to ``path`` and return the updated history."""
    history = load_history(path)
    history.append(record.to_dict())
    save_history(path, history)
    return history


def write_report_json(path: Path, record: BenchmarkRecord) -> None:
    """Write the full benchmark record as ``report.json``."""
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(record.to_dict(), f, indent=2)


def write_metrics_csv(path: Path, record: BenchmarkRecord) -> None:
    """Write a wide CSV with the main and baseline metrics."""
    ensure_dir(path.parent)
    metrics = record.metrics
    baselines = metrics.get("baselines", {})

    fieldnames = [
        "block",
        "model_id",
        "n_masked",
        "cross_entropy",
        "nll",
        "perplexity",
        "cross_entropy_lower_bound",
        "top1_accuracy",
        "top3_accuracy",
        "top5_accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "macro_f1",
        "micro_f1",
        "weighted_f1",
        "mcc",
        "kappa",
        "prediction_entropy",
        "average_confidence",
    ]

    rows: List[Dict[str, Any]] = []
    main_row = {k: metrics.get(k) for k in fieldnames}
    main_row["block"] = record.block
    main_row["model_id"] = record.model_id
    rows.append(main_row)

    for name, baseline in baselines.items():
        row = {k: None for k in fieldnames}
        row["block"] = record.block
        row["model_id"] = f"{record.model_id} (baseline: {name})"
        for k in fieldnames:
            if k in baseline:
                row[k] = baseline[k]
        rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _per_base_rows(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a list of per-base metric rows."""
    rows = []
    per_base = metrics.get("per_base", {})
    for base, values in per_base.items():
        row = {"base": base}
        row.update({k: v for k, v in values.items()})
        rows.append(row)
    return rows


def _format_value(value: Any) -> str:
    """Format a metric value for the printed/ HTML report."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if not (value == value):
            return "N/A"
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
        return f"{value:.4f}"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def _format_percent(value: Any) -> str:
    """Format a 0..1 value as a percentage string."""
    if value is None or not isinstance(value, (int, float)):
        return "N/A"
    return f"{value:.2%}"


def _html_table(rows: List[Dict[str, Any]]) -> str:
    """Build an HTML table from a list of row dicts."""
    if not rows:
        return "<p>No data.</p>"
    headers = list(rows[0].keys())
    lines = ["<table class='metrics-table'>", "<tr>"]
    for h in headers:
        lines.append(f"<th>{html.escape(str(h).replace('_', ' ').title())}</th>")
    lines.append("</tr>")
    for row in rows:
        lines.append("<tr>")
        for h in headers:
            v = row.get(h)
            lines.append(f"<td>{html.escape(_format_value(v))}</td>")
        lines.append("</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def _confusion_html(cm: Dict[str, Dict[str, int]]) -> str:
    """Build an HTML confusion matrix table."""
    labels = sorted(cm.keys())
    lines = ["<table class='metrics-table'>", "<tr><th></th>"]
    for p in labels:
        lines.append(f"<th>{html.escape(p)}</th>")
    lines.append("</tr>")
    for t in labels:
        lines.append(f"<tr><th>{html.escape(t)}</th>")
        for p in labels:
            lines.append(f"<td>{cm[t][p]}</td>")
        lines.append("</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def write_benchmark_html(
    path: Path,
    record: BenchmarkRecord,
    plot_paths: PlotPaths,
) -> None:
    """Write a self-contained ``benchmark.html`` report."""
    ensure_dir(path.parent)
    metrics = record.metrics
    baselines = metrics.get("baselines", {})

    sections: List[str] = []
    sections.append("<h2>Overview</h2>")
    overview = [
        ("Benchmark version", record.benchmark_version),
        ("Timestamp", record.timestamp),
        ("Block", record.block),
        ("Model", record.model_id),
        ("Model version", record.model_version or "N/A"),
        ("Model hash", record.model_hash or "N/A"),
        ("Checkpoint", record.checkpoint or "N/A"),
        ("Genome", record.genome_source),
        ("Genome hash", record.genome_hash or "N/A"),
        ("Seed", record.seed),
        ("Samples", record.num_samples),
        ("Sequence length", record.seq_length),
        ("Mask probability", record.mask_prob),
        ("gRPC address", record.grpc_address),
        ("Vocab size", record.vocab_size or _format_value(metrics.get("vocab_size"))),
        ("Logits available", metrics.get("logits_available")),
        ("Masked positions", metrics.get("n_masked")),
    ]
    sections.append("<table class='metrics-table'>")
    for name, value in overview:
        sections.append(
            f"<tr><th>{html.escape(name)}</th><td>{html.escape(_format_value(value))}</td></tr>"
        )
    sections.append("</table>")

    sections.append("<h2>Core Metrics</h2>")
    core_rows = [
        {
            "Metric": "Cross Entropy",
            "Value": _format_value(metrics.get("cross_entropy")),
        },
        {"Metric": "NLL", "Value": _format_value(metrics.get("nll"))},
        {"Metric": "Perplexity", "Value": _format_value(metrics.get("perplexity"))},
        {"Metric": "Top-1 Accuracy", "Value": _format_percent(metrics.get("top1_accuracy"))},
        {"Metric": "Top-3 Accuracy", "Value": _format_percent(metrics.get("top3_accuracy"))},
        {"Metric": "Top-5 Accuracy", "Value": _format_percent(metrics.get("top5_accuracy"))},
        {
            "Metric": "Balanced Accuracy",
            "Value": _format_percent(metrics.get("balanced_accuracy")),
        },
        {"Metric": "Precision", "Value": _format_percent(metrics.get("precision"))},
        {"Metric": "Recall", "Value": _format_percent(metrics.get("recall"))},
        {"Metric": "Macro F1", "Value": _format_value(metrics.get("macro_f1"))},
        {"Metric": "Micro F1", "Value": _format_value(metrics.get("micro_f1"))},
        {"Metric": "Weighted F1", "Value": _format_value(metrics.get("weighted_f1"))},
        {"Metric": "MCC", "Value": _format_value(metrics.get("mcc"))},
        {"Metric": "Kappa", "Value": _format_value(metrics.get("kappa"))},
        {
            "Metric": "Prediction Entropy",
            "Value": _format_value(metrics.get("prediction_entropy")),
        },
        {
            "Metric": "Average Confidence",
            "Value": _format_value(metrics.get("average_confidence")),
        },
    ]
    sections.append(_html_table(core_rows))

    sections.append("<h2>Per-Base Metrics</h2>")
    sections.append(_html_table(_per_base_rows(metrics)))

    sections.append("<h2>Baselines</h2>")
    baseline_rows = []
    for name, baseline in baselines.items():
        row = {"Baseline": name.title()}
        row.update(
            {
                "Accuracy": _format_percent(baseline.get("accuracy")),
                "Top-3": _format_percent(baseline.get("top3_accuracy")),
                "Top-5": _format_percent(baseline.get("top5_accuracy")),
                "Macro F1": _format_value(baseline.get("macro_f1")),
                "Perplexity": _format_value(baseline.get("perplexity")),
                "Average Confidence": _format_value(baseline.get("average_confidence")),
            }
        )
        note = baseline.get("note")
        if note:
            row["Note"] = note
        baseline_rows.append(row)
    sections.append(_html_table(baseline_rows))

    sections.append("<h2>Confusion Matrix</h2>")
    sections.append(_confusion_html(metrics.get("confusion_matrix", {})))

    if record.region_metrics:
        sections.append("<h2>Genomic Region Metrics</h2>")
        region_rows = []
        for region, rmetrics in record.region_metrics.items():
            row = {
                "Region": region,
                "N masked": rmetrics.get("n_masked"),
                "Top-1": _format_percent(rmetrics.get("top1_accuracy")),
                "Balanced Acc": _format_percent(rmetrics.get("balanced_accuracy")),
                "Macro F1": _format_value(rmetrics.get("macro_f1")),
                "Perplexity": _format_value(rmetrics.get("perplexity")),
            }
            region_rows.append(row)
        sections.append(_html_table(region_rows))

    sections.append("<h2>Plots</h2>")
    plot_dir = path.parent
    for attr in ["accuracy", "loss", "perplexity", "confidence", "confusion_matrix", "prediction_distribution", "training_history"]:
        p = getattr(plot_paths, attr)
        if p and p.exists():
            rel = os.path.relpath(p, path.parent)
            sections.append(f"<h3>{attr.replace('_', ' ').title()}</h3>")
            sections.append(f"<img src='{html.escape(rel)}' alt='{html.escape(attr)}'>")

    if record.warnings:
        sections.append("<h2>Warnings</h2>")
        sections.append("<ul>")
        for w in record.warnings:
            sections.append(f"<li>{html.escape(w)}</li>")
        sections.append("</ul>")

    style = """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem; color: #222; }
      h1, h2, h3 { color: #1a1a1a; }
      .metrics-table { border-collapse: collapse; margin: 1rem 0; width: 100%; max-width: 700px; }
      .metrics-table th, .metrics-table td { border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; }
      .metrics-table th { background: #f5f5f5; }
      img { max-width: 100%; height: auto; margin: 1rem 0; }
      ul { max-width: 700px; }
    </style>
    """

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Xenomorph MLM Benchmark - Block {record.block}</title>
{style}
</head>
<body>
<h1>Xenomorph MLM Benchmark</h1>
{"\n".join(sections)}
</body>
</html>"""

    with open(path, "w") as f:
        f.write(html_doc)


def write_outputs(
    record: BenchmarkRecord,
    output_dir: Path,
    history_path: Optional[Path] = None,
    generate_plots_flag: bool = True,
) -> List[str]:
    """Write all benchmark output files and return the list of file paths."""
    ensure_dir(output_dir)

    report_path = output_dir / "report.json"
    metrics_path = output_dir / "metrics.csv"
    html_path = output_dir / "benchmark.html"
    history_file = history_path or (output_dir / "history.json")

    write_report_json(report_path, record)
    write_metrics_csv(metrics_path, record)

    if generate_plots_flag:
        plot_paths = generate_plots(record.metrics, output_dir)
        history = append_history(history_file, record)
        generate_history_plot(history, output_dir)
    else:
        plot_paths = PlotPaths()
        history = append_history(history_file, record)

    write_benchmark_html(html_path, record, plot_paths)

    record.output_files = [
        str(report_path),
        str(metrics_path),
        str(html_path),
        str(history_file),
    ]
    if plot_paths.accuracy:
        record.output_files.append(str(plot_paths.accuracy))
    if plot_paths.loss:
        record.output_files.append(str(plot_paths.loss))
    if plot_paths.perplexity:
        record.output_files.append(str(plot_paths.perplexity))
    if plot_paths.confidence:
        record.output_files.append(str(plot_paths.confidence))
    if plot_paths.confusion_matrix:
        record.output_files.append(str(plot_paths.confusion_matrix))
    if plot_paths.prediction_distribution:
        record.output_files.append(str(plot_paths.prediction_distribution))
    if plot_paths.training_history:
        record.output_files.append(str(plot_paths.training_history))

    write_report_json(report_path, record)  # update with output_files
    return record.output_files
