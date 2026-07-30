#!/usr/bin/env python3
"""Xenomorph MLM Benchmark.

Scientific reference evaluation for Masked Language Models served by the
Xenomorph UsefulPoW blockchain.

Example:
    python scripts/evaluate_mlm_benchmark.py \
        --model xeno/mgm-1 \
        --block 2500 \
        --genome ~/.rusty-xenom/grch38.xenom \
        --seed 42 \
        --batch-size 16 \
        --plots \
        --output benchmark.html
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

# Allow importing `xenom_benchmark` from the repository root and the generated
# gRPC stubs from `scripts/xenom_eval`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from xenom_benchmark.evaluator import Evaluator, EvaluatorConfig
from xenom_benchmark.grpc_backend import GrpcBackend, ModelInfo
from xenom_benchmark.report import BenchmarkRecord, write_outputs, write_report_json
from xenom_benchmark.utils import DEFAULT_GRPC_ADDR, setup_logging


def parse_blocks(value: str) -> List[int]:
    """Parse a comma-separated list of block numbers."""
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def resolve_output_paths(
    output: str,
    json_path: Optional[str],
    csv_path: Optional[str],
    history_path: Optional[str],
) -> tuple:
    """Resolve output directory and file paths from --output and optional overrides."""
    out = Path(output)
    if out.suffix.lower() == ".html":
        output_dir = out.parent
        html_path = out
    else:
        output_dir = out
        html_path = output_dir / "benchmark.html"

    output_dir = output_dir.resolve()
    if json_path:
        report_json = Path(json_path).resolve()
    else:
        report_json = output_dir / "report.json"

    if csv_path:
        metrics_csv = Path(csv_path).resolve()
    else:
        metrics_csv = output_dir / "metrics.csv"

    if history_path:
        history_json = Path(history_path).resolve()
    else:
        history_json = output_dir / "history.json"

    html_path = html_path.resolve()

    return output_dir, html_path, report_json, metrics_csv, history_json


def print_summary(record: BenchmarkRecord, width: int = 72) -> None:
    """Print a human-readable benchmark summary to stdout."""
    metrics = record.metrics
    baselines = metrics.get("baselines", {})

    def fmt(value):
        if value is None:
            return "N/A"
        if isinstance(value, float):
            if value == float("inf"):
                return "inf"
            if value == float("-inf"):
                return "-inf"
            return f"{value:.4f}"
        return str(value)

    print("=" * width)
    print("MGM-1 Benchmark".center(width))
    print("=" * width)
    print(f"Model            : {record.model_id}")
    print(f"Checkpoint       : {record.model_hash or 'N/A'}")
    print(f"Block            : {record.block}")
    print(f"Hash             : {record.model_hash or 'N/A'}")
    print(f"Genome           : {record.genome_source}")
    print(f"Mask Probability : {record.mask_prob}")
    print(f"Samples          : {record.num_samples}")
    print("-" * width)
    print(f"Loss (CE)        : {fmt(metrics.get('cross_entropy'))}")
    print(f"Cross Entropy    : {fmt(metrics.get('cross_entropy'))}")
    print(f"NLL              : {fmt(metrics.get('nll'))}")
    print(f"Perplexity       : {fmt(metrics.get('perplexity'))}")
    print(f"Top-1            : {fmt(metrics.get('top1_accuracy'))}")
    print(f"Top-3            : {fmt(metrics.get('top3_accuracy'))}")
    print(f"Top-5            : {fmt(metrics.get('top5_accuracy'))}")
    print(f"Balanced Accuracy: {fmt(metrics.get('balanced_accuracy'))}")
    print(f"Macro F1         : {fmt(metrics.get('macro_f1'))}")
    print(f"Micro F1         : {fmt(metrics.get('micro_f1'))}")
    print(f"Weighted F1      : {fmt(metrics.get('weighted_f1'))}")
    print(f"MCC              : {fmt(metrics.get('mcc'))}")
    print(f"Kappa            : {fmt(metrics.get('kappa'))}")
    print(f"Prediction Entropy: {fmt(metrics.get('prediction_entropy'))}")
    print(f"Average Confidence: {fmt(metrics.get('average_confidence'))}")
    print("-" * width)
    print("Per-base metrics")
    per_base = metrics.get("per_base", {})
    for base in sorted(per_base.keys()):
        pb = per_base[base]
        print(
            f"  {base}: acc={fmt(pb.get('accuracy'))} "
            f"p={fmt(pb.get('precision'))} r={fmt(pb.get('recall'))} f1={fmt(pb.get('f1'))}"
        )
    print("-" * width)
    print("Baselines")
    for name, baseline in baselines.items():
        print(
            f"  {name:12s}: acc={fmt(baseline.get('accuracy'))} "
            f"ppl={fmt(baseline.get('perplexity'))} "
            f"conf={fmt(baseline.get('average_confidence'))}"
        )
    print("-" * width)
    print("Prediction Distribution")
    pred_dist = metrics.get("prediction_distribution", {})
    total = sum(pred_dist.values()) if pred_dist else 1
    parts = [
        f"{b}: {pred_dist.get(b, 0)} ({pred_dist.get(b, 0) / total:.2%})"
        for b in sorted(pred_dist.keys())
    ]
    print("  " + ", ".join(parts))
    print("-" * width)
    if record.warnings:
        print("Warnings")
        for w in record.warnings[:10]:
            print(f"  - {w}")
        if len(record.warnings) > 10:
            print(f"  ... and {len(record.warnings) - 10} more")
    print("-" * width)
    print("Output files")
    for p in record.output_files:
        print(f"  {p}")
    print("=" * width)


def run_single_block(
    backend: GrpcBackend,
    args: argparse.Namespace,
    block: int,
    output_dir: Path,
    report_json: Path,
    metrics_csv: Path,
    html_path: Path,
    history_json: Path,
) -> BenchmarkRecord:
    """Run the benchmark for a single block and write outputs."""
    num_samples = args.num_samples if args.num_samples else args.batch_size

    config = EvaluatorConfig(
        model_id=args.model,
        block=block,
        genome=args.genome,
        seed=args.seed,
        num_samples=num_samples,
        seq_length=args.seq_length,
        mask_prob=args.mask_prob,
        batch_size=args.batch_size,
        grpc_address=args.grpc_address,
        annotation=args.annotation,
        vocab_size=args.vocab_size,
        timeout=args.timeout,
        tokenizer_path=Path(args.tokenizer_path) if args.tokenizer_path else None,
        use_token_level=args.token_level,
    )

    evaluator = Evaluator(backend, config)
    record = evaluator.run()

    # Add any user-supplied block metadata to the gRPC call for future backends.
    record.grpc_address = args.grpc_address

    output_files = write_outputs(
        record,
        output_dir,
        history_path=history_json,
        generate_plots_flag=args.plots,
    )

    # Rename report.json and metrics.csv if custom paths were requested.
    if report_json != output_dir / "report.json" and (output_dir / "report.json").exists():
        (output_dir / "report.json").rename(report_json)
    if metrics_csv != output_dir / "metrics.csv" and (output_dir / "metrics.csv").exists():
        (output_dir / "metrics.csv").rename(metrics_csv)
    if html_path != output_dir / "benchmark.html" and (output_dir / "benchmark.html").exists():
        (output_dir / "benchmark.html").rename(html_path)

    # Update the record with the final output paths and rewrite report.json.
    record.output_files = [
        str(report_json),
        str(metrics_csv),
        str(html_path),
        str(history_json),
    ] + [str(p) for p in output_files[4:] if p]
    write_report_json(report_json, record)

    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Xenomorph MLM benchmark for blockchain-served models."
    )
    parser.add_argument("--model", default="xeno/mgm-1", help="Model id to evaluate.")
    parser.add_argument(
        "--block", type=int, default=0, help="Block number to tag this evaluation."
    )
    parser.add_argument(
        "--blocks",
        type=parse_blocks,
        help="Comma-separated list of block numbers to produce a learning curve.",
    )
    parser.add_argument("--genome", help="Path to .xenom or .fasta genome source.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of samples per batch; used as num-samples if num-samples is not set.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Total number of test sequences (overrides --batch-size).",
    )
    parser.add_argument(
        "--seq-length", type=int, default=512, help="Length of each test sequence."
    )
    parser.add_argument(
        "--mask-prob", type=float, default=0.15, help="Fraction of positions to mask."
    )
    parser.add_argument(
        "--output", default="benchmark.html", help="Output HTML path or directory."
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        help="Path for report.json (default: <output_dir>/report.json).",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        help="Path for metrics.csv (default: <output_dir>/metrics.csv).",
    )
    parser.add_argument(
        "--history",
        dest="history_path",
        help="Path for history.json (default: <output_dir>/history.json).",
    )
    parser.add_argument(
        "--plots", action="store_true", help="Generate PNG plots."
    )
    parser.add_argument(
        "--grpc-address",
        default=DEFAULT_GRPC_ADDR,
        help=f"gRPC inference server address (default: {DEFAULT_GRPC_ADDR}).",
    )
    parser.add_argument(
        "--annotation",
        help="Optional GFF/GTF/BED annotation file for region-split metrics.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        help="Override vocabulary size for no-logit approximations.",
    )
    parser.add_argument(
        "--tokenizer-path",
        help="Path to a local tokenizer.json. If omitted and the model is not "
        "MGM-1, the tokenizer is downloaded from HuggingFace.",
    )
    parser.add_argument(
        "--token-level",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use token-level masking for BPE models (default: on).",
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="gRPC call timeout in seconds."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging."
    )

    args = parser.parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    blocks = args.blocks if args.blocks else [args.block]

    output_dir, html_path, report_json, metrics_csv, history_json = resolve_output_paths(
        args.output, args.json_path, args.csv_path, args.history_path
    )

    with GrpcBackend(args.grpc_address, timeout=args.timeout) as backend:
        previous_hash: Optional[str] = None
        records: List[BenchmarkRecord] = []

        for i, block in enumerate(blocks):
            print(f"\n[Block {block}] Running benchmark ...")
            try:
                info = backend.get_model_info(args.model)
                current_hash = info.model_hash
            except Exception as exc:
                current_hash = None
                logging.warning("Could not retrieve model hash: %s", exc)

            if i > 0 and current_hash and previous_hash and current_hash == previous_hash:
                logging.warning(
                    "Model hash did not change between block %s and block %s; "
                    "the learning curve will be degenerate unless the server "
                    "switched checkpoints.",
                    blocks[i - 1],
                    block,
                )

            record = run_single_block(
                backend,
                args,
                block,
                output_dir,
                report_json,
                metrics_csv,
                html_path,
                history_json,
            )
            records.append(record)
            print_summary(record)

            if current_hash:
                previous_hash = current_hash

    return 0


if __name__ == "__main__":
    sys.exit(main())
