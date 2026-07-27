#!/usr/bin/env python3
"""
evaluate_mgm1_training.py

Validates the training progress of the xeno/mgm-1 model by evaluating it on a
fixed set of real DNA sequences from a GRCh38 .xenom archive.

The script follows the same reporting format as the Real Genome MLM Monitor:
    - Perplexity (confidence-based proxy)
    - Token accuracy
    - Accuracy by base (A, T, C, G)
    - Prediction distribution
    - Average confidence
    - Trend vs previous evaluation

It can also run in --diagnostic mode to verify that the seed-node is producing
genome batches from different regions of the archive.

Requirements:
    pip install grpcio protobuf requests websockets

If the generated gRPC stubs are not present, run from the repo root:
    python -m grpc_tools.protoc \
        --python_out=scripts/xenom_eval \
        --grpc_python_out=scripts/xenom_eval \
        -Iproto proto/inference.proto

Example:
    # Evaluate the model currently loaded on the local seed-node
    python scripts/evaluate_mgm1_training.py --block 50 --genome grch38.xenom

    # Diagnostic: check that WebSocket genome batches vary
    python scripts/evaluate_mgm1_training.py --diagnostic --ws-url ws://127.0.0.1:17110
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import struct
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Generated gRPC stubs live in scripts/xenom_eval/
SCRIPT_DIR = Path(__file__).resolve().parent
GRPC_STUBS_DIR = SCRIPT_DIR / "xenom_eval"
sys.path.insert(0, str(GRPC_STUBS_DIR))

try:
    import grpc
    import inference_pb2, inference_pb2_grpc
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Missing Python dependencies or generated gRPC stubs.\n"
        f"Error: {e}\n"
        "Install: pip install grpcio protobuf requests websockets\n"
        "Generate stubs: python -m grpc_tools.protoc "
        "--python_out=scripts/xenom_eval --grpc_python_out=scripts/xenom_eval "
        "-Iproto proto/inference.proto"
    )

# Canonical GRCh38 merkle root used by the miner when no --genome-merkle is given.
DEFAULT_GENOME_MERKLE = "577126c448d24d132ba77436517a7db2203d6fce0cd81e2b84db39875d43ee80"

DEFAULT_GRPC_ADDR = "127.0.0.1:50051"
DEFAULT_WS_URL = "ws://127.0.0.1:17110"
DEFAULT_MODEL_ID = "xeno/mgm-1"
DEFAULT_SEQ_LEN = 512
DEFAULT_MASK_RATIO = 0.15
DEFAULT_N_SEQUENCES = 88
DEFAULT_FRAGMENT_SIZE = 1_048_576
GENOME_DOWNLOAD_URL = "https://github.com/hainakus/Xenomorph/releases/download/genome-grch38-v0/grch38.xenom"

BASES = ("A", "T", "C", "G")
MASK_CHARS = ("[", "<mask>", "[MASK]")


# -----------------------------------------------------------------------------
# .xenom archive parser (matches seed-node/src/genome/archive.rs)
# -----------------------------------------------------------------------------

class XenomArchive:
    """In-memory parser for a XENOGEN1 2-bit packed genome archive."""

    BITS_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T"}
    BASE_TO_BITS = {v: k for k, v in BITS_TO_BASE.items()}

    def __init__(self, path: str, fragment_size: int = DEFAULT_FRAGMENT_SIZE):
        self.path = Path(path)
        self.fragment_size = fragment_size
        with self.path.open("rb") as f:
            header_bytes = f.read(64)
            if len(header_bytes) != 64:
                raise ValueError(f"{path}: file too small for 64-byte header")

            self.magic = header_bytes[:8]
            if self.magic != b"XENOGEN1":
                raise ValueError(f"{path}: invalid magic {self.magic!r}, expected XENOGEN1")

            self.version = struct.unpack_from("<I", header_bytes, 8)[0]
            self.dataset_version = struct.unpack_from("<I", header_bytes, 12)[0]
            self.total_bases = struct.unpack_from("<Q", header_bytes, 16)[0]
            self.total_packed_bytes = struct.unpack_from("<Q", header_bytes, 24)[0]
            self.merkle_root = header_bytes[32:64].hex()

            self.data = f.read()
            if len(self.data) != self.total_packed_bytes:
                raise ValueError(
                    f"{path}: packed data size mismatch "
                    f"(expected {self.total_packed_bytes}, got {len(self.data)})"
                )

    def num_fragments(self) -> int:
        packed_frag = self.fragment_size // 4
        if packed_frag == 0:
            return 0
        return self.total_packed_bytes // packed_frag

    def fragment_base_count(self, idx: int) -> int:
        start = idx * self.fragment_size
        remaining = self.total_bases - start
        return min(self.fragment_size, remaining)

    def extract_sequence(self, fragment_idx: int, start: int, length: int) -> str:
        fragment_bases = self.fragment_base_count(fragment_idx)
        if start + length > fragment_bases:
            raise ValueError(
                f"Range {start}..{start + length} exceeds fragment size {fragment_bases}"
            )

        packed_frag = self.fragment_size // 4
        offset = fragment_idx * packed_frag
        end = min(offset + packed_frag, len(self.data))
        packed = self.data[offset:end]

        seq = []
        for i in range(length):
            base_offset = start + i
            byte_idx = base_offset // 4
            shift = 6 - 2 * (base_offset % 4)
            bits = (packed[byte_idx] >> shift) & 0b11
            seq.append(self.BITS_TO_BASE[bits])
        return "".join(seq)

    def random_sequences(self, n: int, length: int, seed: int) -> List[str]:
        """Sample n random non-overlapping windows from the archive."""
        rng = random.Random(seed)
        sequences = []
        attempts = 0
        while len(sequences) < n and attempts < n * 100:
            attempts += 1
            fragment_idx = rng.randrange(self.num_fragments())
            fragment_bases = self.fragment_base_count(fragment_idx)
            if fragment_bases < length:
                continue
            max_start = fragment_bases - length
            start = rng.randrange(0, max_start + 1)
            try:
                seq = self.extract_sequence(fragment_idx, start, length)
            except (ValueError, IndexError):
                continue
            # Skip windows that contain non-ACGT bases (shouldn't happen, but be safe).
            if set(seq).issubset(BASES):
                sequences.append(seq)

        if len(sequences) < n:
            raise RuntimeError(
                f"Could only extract {len(sequences)}/{n} valid {length}-bp sequences "
                f"from {self.path}"
            )
        return sequences


# -----------------------------------------------------------------------------
# FASTA loader (fallback / external benchmark sets)
# -----------------------------------------------------------------------------

def load_fasta(path: str, min_len: int = 32) -> List[str]:
    """Return a list of uppercase ACGT sequences from a FASTA file."""
    seqs = []
    current = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    s = "".join(current).upper()
                    if len(s) >= min_len and set(s).issubset(BASES):
                        seqs.append(s)
                    current = []
            else:
                current.append(line)
    if current:
        s = "".join(current).upper()
        if len(s) >= min_len and set(s).issubset(BASES):
            seqs.append(s)
    return seqs


def sample_fasta_sequences(path: str, n: int, length: int, seed: int) -> List[str]:
    """Sample n random length-bp windows from FASTA sequences."""
    all_seqs = load_fasta(path)
    if not all_seqs:
        raise ValueError(f"No valid sequences found in {path}")

    concatenated = "".join(all_seqs)
    rng = random.Random(seed)
    seqs = []
    attempts = 0
    while len(seqs) < n and attempts < n * 100:
        attempts += 1
        start = rng.randrange(0, max(1, len(concatenated) - length + 1))
        window = concatenated[start : start + length]
        if len(window) == length and set(window).issubset(BASES):
            seqs.append(window)
    if len(seqs) < n:
        raise RuntimeError(f"Could only extract {len(seqs)}/{n} windows from {path}")
    return seqs


# -----------------------------------------------------------------------------
# gRPC inference client
# -----------------------------------------------------------------------------

class GrpcInferenceClient:
    def __init__(self, addr: str = DEFAULT_GRPC_ADDR):
        self.addr = addr
        self.channel = grpc.insecure_channel(addr)
        self.stub = inference_pb2_grpc.InferenceStub(self.channel)

    def close(self):
        self.channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def get_model_info(self, model_id: str) -> inference_pb2.ModelInfoResponse:
        return self.stub.GetModelInfo(inference_pb2.ModelInfoRequest(model_id=model_id))

    def predict(self, model_id: str, input_data: str) -> inference_pb2.PredictResponse:
        return self.stub.Predict(
            inference_pb2.PredictRequest(
                model_id=model_id,
                input_data=input_data.encode(),
                query_id=f"eval-{int(time.time() * 1000)}",
            )
        )


# -----------------------------------------------------------------------------
# Minimal Borsh codec for the WebSocket RPC used by the miner
# -----------------------------------------------------------------------------

class BorshCodec:
    """Tiny Borsh encoder/decoder for the messages used by this script."""

    U8 = 0
    U16 = 1
    U32 = 2
    U64 = 3
    I8 = 4
    I16 = 5
    I32 = 6
    I64 = 7
    F32 = 8
    F64 = 9
    STRING = 10

    @staticmethod
    def pack_u8(v: int) -> bytes:
        return struct.pack("<B", v)

    @staticmethod
    def pack_u32(v: int) -> bytes:
        return struct.pack("<I", v)

    @staticmethod
    def pack_u64(v: int) -> bytes:
        return struct.pack("<Q", v)

    @staticmethod
    def pack_f32(v: float) -> bytes:
        return struct.pack("<f", v)

    @staticmethod
    def pack_string(s: str) -> bytes:
        b = s.encode("utf-8")
        return BorshCodec.pack_u32(len(b)) + b

    @staticmethod
    def pack_bytes32(b: bytes) -> bytes:
        if len(b) != 32:
            raise ValueError("bytes32 must be exactly 32 bytes")
        return b

    @staticmethod
    def pack_vec(items: List[bytes]) -> bytes:
        return BorshCodec.pack_u32(len(items)) + b"".join(items)

    @staticmethod
    def pack_genome_slice(gs: Dict) -> bytes:
        return (
            BorshCodec.pack_u64(gs["chunk_idx"])
            + BorshCodec.pack_u32(gs["start_base"])
            + BorshCodec.pack_u32(gs["length"])
        )

    @staticmethod
    def pack_genome_training_batch(req: Dict) -> bytes:
        return (
            BorshCodec.pack_bytes32(bytes.fromhex(req["genome_merkle_root"]))
            + BorshCodec.pack_string(req["model_id"])
            + BorshCodec.pack_u64(req["preferred_batch_size"])
        )

    @staticmethod
    def pack_envelope(request_id: int, payload: bytes, variant: int) -> bytes:
        # RpcRequest enum: variant index (u8) + payload
        return BorshCodec.pack_u64(request_id) + BorshCodec.pack_u8(variant) + payload

    @staticmethod
    def unpack_u8(data: bytes, off: int) -> Tuple[int, int]:
        return data[off], off + 1

    @staticmethod
    def unpack_u32(data: bytes, off: int) -> Tuple[int, int]:
        return struct.unpack_from("<I", data, off)[0], off + 4

    @staticmethod
    def unpack_u64(data: bytes, off: int) -> Tuple[int, int]:
        return struct.unpack_from("<Q", data, off)[0], off + 8

    @staticmethod
    def unpack_f32(data: bytes, off: int) -> Tuple[float, int]:
        return struct.unpack_from("<f", data, off)[0], off + 4

    @staticmethod
    def unpack_string(data: bytes, off: int) -> Tuple[str, int]:
        length, off = BorshCodec.unpack_u32(data, off)
        return data[off : off + length].decode("utf-8"), off + length

    @staticmethod
    def unpack_bytes(data: bytes, off: int, length: int) -> Tuple[bytes, int]:
        return data[off : off + length], off + length

    @staticmethod
    def unpack_genome_slice(data: bytes, off: int) -> Tuple[Dict, int]:
        chunk_idx, off = BorshCodec.unpack_u64(data, off)
        start_base, off = BorshCodec.unpack_u32(data, off)
        length, off = BorshCodec.unpack_u32(data, off)
        return {"chunk_idx": chunk_idx, "start_base": start_base, "length": length}, off

    @staticmethod
    def unpack_genome_training_batch(data: bytes, off: int) -> Tuple[Dict, int]:
        merkle, off = BorshCodec.unpack_bytes(data, off, 32)
        model_id, off = BorshCodec.unpack_string(data, off)
        preferred, off = BorshCodec.unpack_u64(data, off)
        return {
            "genome_merkle_root": merkle.hex(),
            "model_id": model_id,
            "preferred_batch_size": preferred,
        }, off

    @staticmethod
    def unpack_genome_training_batch_msg(data: bytes, off: int) -> Tuple[Dict, int]:
        # GenomeTrainingBatch
        batch_id, off = BorshCodec.unpack_u64(data, off)
        model_id, off = BorshCodec.unpack_string(data, off)
        merkle, off = BorshCodec.unpack_bytes(data, off, 32)

        count, off = BorshCodec.unpack_u32(data, off)
        data_indices = []
        for _ in range(count):
            gs, off = BorshCodec.unpack_genome_slice(data, off)
            data_indices.append(gs)

        mask_ratio, off = BorshCodec.unpack_f32(data, off)
        seq_length, off = BorshCodec.unpack_u64(data, off)

        batch = {
            "batch_id": batch_id,
            "model_id": model_id,
            "genome_merkle_root": merkle.hex(),
            "data_indices": data_indices,
            "mask_ratio": mask_ratio,
            "seq_length": seq_length,
        }

        # Vec<String> sequences
        count, off = BorshCodec.unpack_u32(data, off)
        sequences = []
        for _ in range(count):
            s, off = BorshCodec.unpack_string(data, off)
            sequences.append(s)

        base_checkpoint, off = BorshCodec.unpack_bytes(data, off, 32)
        return {
            "batch": batch,
            "sequences": sequences,
            "base_checkpoint": base_checkpoint.hex(),
        }, off

    @staticmethod
    def unpack_rpc_response(data: bytes) -> Dict:
        tag, off = BorshCodec.unpack_u8(data, 0)
        # RpcResponse variant indices (must match seed-node/src/rpc/messages.rs):
        # 0 TrainingBatch, 1 ModelCheckpoint, 2 BlockHash, 3 Balance,
        # 4 Difficulty, 5 Pong, 6 Error, 7 GenomeTrainingBatch, ...
        if tag == 7:
            msg, _ = BorshCodec.unpack_genome_training_batch_msg(data, off)
            return {"type": "GenomeTrainingBatch", **msg}
        if tag == 6:
            text, _ = BorshCodec.unpack_string(data, off)
            return {"type": "Error", "message": text}
        return {"type": f"Unknown({tag})"}


# -----------------------------------------------------------------------------
# WebSocket batch diagnostic
# -----------------------------------------------------------------------------

class WebsocketRpcClient:
    def __init__(self, url: str = DEFAULT_WS_URL):
        self.url = url

    async def get_genome_training_batch(
        self,
        genome_merkle_root: str,
        model_id: str,
        preferred_batch_size: int,
        request_id: int = 1,
    ) -> Dict:
        try:
            import websockets
        except ImportError as e:
            raise SystemExit(f"Missing websockets library: {e}")

        payload = BorshCodec.pack_genome_training_batch(
            {
                "genome_merkle_root": genome_merkle_root,
                "model_id": model_id,
                "preferred_batch_size": preferred_batch_size,
            }
        )
        # RpcRequest::GetGenomeTrainingBatch is variant index 6
        envelope = BorshCodec.pack_envelope(request_id, payload, 6)

        async with websockets.connect(self.url) as ws:
            await ws.send(envelope)
            data = await ws.recv()
            return BorshCodec.unpack_rpc_response(data)


# -----------------------------------------------------------------------------
# Core evaluation logic
# -----------------------------------------------------------------------------

class Mgm1Evaluator:
    def __init__(
        self,
        grpc_client: GrpcInferenceClient,
        model_id: str = DEFAULT_MODEL_ID,
        seq_len: int = DEFAULT_SEQ_LEN,
        mask_ratio: float = DEFAULT_MASK_RATIO,
    ):
        self.grpc = grpc_client
        self.model_id = model_id
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio

    def build_test_set(
        self, source: str, n: int = DEFAULT_N_SEQUENCES, seed: int = 42
    ) -> List[str]:
        path = Path(source)
        if path.suffix.lower() == ".xenom":
            archive = XenomArchive(source)
            return archive.random_sequences(n, self.seq_len, seed)
        if path.suffix.lower() in (".fasta", ".fa"):
            return sample_fasta_sequences(source, n, self.seq_len, seed)
        raise ValueError(f"Unsupported genome source: {source} (use .xenom or .fasta)")

    @staticmethod
    def mask_sequence(seq: str, mask_ratio: float, seed: Optional[int]) -> Tuple[str, List[int]]:
        rng = random.Random(seed)
        indices = [i for i in range(len(seq)) if rng.random() < mask_ratio]
        masked = list(seq)
        for i in indices:
            masked[i] = "["
        return "".join(masked), indices

    def evaluate(self, test_set: List[str], test_seed: int = 42) -> Dict:
        results = []
        for seq_idx, original in enumerate(test_set):
            masked, mask_positions = self.mask_sequence(
                original, self.mask_ratio, test_seed + seq_idx
            )
            if not mask_positions:
                continue

            response = self.grpc.predict(self.model_id, masked)
            predicted = response.output_data.decode("utf-8")

            seq_result = self._analyse_sequence(
                original, masked, predicted, mask_positions, response.confidence
            )
            results.append(seq_result)

        return self._aggregate(results)

    def _analyse_sequence(
        self,
        original: str,
        masked: str,
        predicted: str,
        mask_positions: List[int],
        confidence: float,
    ) -> Dict:
        if len(predicted) != len(original):
            print(
                f"[WARN] Length mismatch for a sequence: "
                f"expected {len(original)}, got {len(predicted)}"
            )

        correct = 0
        per_base = {b: {"total": 0, "correct": 0} for b in BASES}
        predictions = []

        for pos in mask_positions:
            true_base = original[pos] if pos < len(original) else "?"
            pred_base = predicted[pos] if pos < len(predicted) else "?"
            predictions.append(pred_base)
            per_base[true_base]["total"] += 1
            if pred_base in BASES and pred_base == true_base:
                correct += 1
                per_base[true_base]["correct"] += 1

        return {
            "mask_count": len(mask_positions),
            "correct": correct,
            "per_base": per_base,
            "predictions": predictions,
            "confidence": confidence,
            "valid_output": set(predicted).issubset(BASES),
        }

    def _aggregate(self, results: List[Dict]) -> Dict:
        total_masks = sum(r["mask_count"] for r in results)
        total_correct = sum(r["correct"] for r in results)
        token_accuracy = total_correct / total_masks if total_masks else 0.0

        per_base = {b: {"total": 0, "correct": 0} for b in BASES}
        for r in results:
            for b in BASES:
                per_base[b]["total"] += r["per_base"][b]["total"]
                per_base[b]["correct"] += r["per_base"][b]["correct"]

        base_accuracy = {}
        for b in BASES:
            t = per_base[b]["total"]
            c = per_base[b]["correct"]
            base_accuracy[b] = c / t if t else 0.0

        all_predictions = []
        for r in results:
            for p in r["predictions"]:
                if p in BASES:
                    all_predictions.append(p)
        pred_dist = {b: Counter(all_predictions).get(b, 0) for b in BASES}

        # Weighted average confidence across masked positions (each sequence
        # contributes confidence * mask_count because the gRPC confidence is the
        # average probability at every masked position in that sequence).
        weighted_conf = sum(r["confidence"] * r["mask_count"] for r in results)
        avg_confidence = weighted_conf / total_masks if total_masks else 0.0

        # Confidence-based perplexity (geometric mean of argmax probabilities).
        # This is the same proxy used by the Real Genome MLM Monitor.
        if avg_confidence > 0 and total_masks:
            log_sum = sum(
                math.log(max(r["confidence"], 1e-12)) * r["mask_count"]
                for r in results
            )
            perplexity = math.exp(-log_sum / total_masks)
        else:
            perplexity = float("inf")

        invalid_outputs = sum(1 for r in results if not r["valid_output"])

        return {
            "total_sequences": len(results),
            "total_masks": total_masks,
            "token_accuracy": token_accuracy,
            "per_base_accuracy": base_accuracy,
            "prediction_distribution": pred_dist,
            "average_confidence": avg_confidence,
            "perplexity": perplexity,
            "invalid_outputs": invalid_outputs,
        }


# -----------------------------------------------------------------------------
# Trend / history helpers
# -----------------------------------------------------------------------------

def load_history(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def save_history(path: str, history: List[Dict]) -> None:
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def compute_trend(current: Dict, previous: Optional[Dict]) -> str:
    if not previous:
        return "initial"
    acc_delta = current["token_accuracy"] - previous["token_accuracy"]
    ppl_delta = current["perplexity"] - previous["perplexity"]

    # Lower perplexity is better; higher accuracy is better.
    if ppl_delta < -0.005 and acc_delta > 0.005:
        return "improving"
    if ppl_delta > 0.005 and acc_delta < -0.005:
        return "degrading"
    return "stable"


def validate_metrics(metrics: Dict, warn_threshold: float = 0.05) -> List[str]:
    warnings = []
    if metrics["token_accuracy"] <= warn_threshold:
        warnings.append(
            f"Token accuracy {metrics['token_accuracy']:.2%} is at or below random guessing "
            f"for 4-class MLM."
        )
    if metrics["invalid_outputs"] > 0:
        warnings.append(
            f"{metrics['invalid_outputs']} sequences produced non-DNA characters."
        )
    if metrics["perplexity"] > 2.5:
        warnings.append(
            f"Perplexity {metrics['perplexity']:.3f} is high (4-class uniform = 4.0)."
        )
    accs = [metrics["per_base_accuracy"][b] for b in BASES]
    if max(accs) - min(accs) > 0.15:
        warnings.append(
            "Imbalanced accuracy across bases: "
            + " | ".join(f"{b}: {metrics['per_base_accuracy'][b]:.2%}" for b in BASES)
        )
    return warnings


# -----------------------------------------------------------------------------
# Report formatting
# -----------------------------------------------------------------------------

def print_report(metrics: Dict, block: int, model_id: str, trend: str, warnings: List[str]):
    width = 70
    print("=" * width)
    print(f" Real Genome MLM Evaluation - Block {block}")
    print("=" * width)
    print(f" Model: {model_id}")
    print(f" Test set: {metrics['total_sequences']} sequences, "
          f"{DEFAULT_MASK_RATIO:.0%} masked")
    print(f" Total masked positions: {metrics['total_masks']}")
    print("-" * width)
    print(" Core Metrics:")
    print(f"   Perplexity:      {metrics['perplexity']:.3f} (lower is better)")
    # correct count
    correct = int(round(metrics['token_accuracy'] * metrics['total_masks']))
    print(f"   Token Accuracy:  {metrics['token_accuracy']:.2%} ({correct}/{metrics['total_masks']})")
    print("-" * width)
    print(" Accuracy by Base:")
    print(
        " | ".join(
            f"{b}: {metrics['per_base_accuracy'][b]:.2%}"
            for b in BASES
        )
    )
    print("-" * width)
    print(" Prediction Distribution:")
    dist = metrics["prediction_distribution"]
    print(
        ", ".join(
            f"'{b}': {dist.get(b, 0)}" for b in BASES
        )
    )
    print("-" * width)
    print(f" Average Confidence: {metrics['average_confidence']:.3f}")
    print(f" Trend: {trend}")
    if warnings:
        print("-" * width)
        print(" Warnings:")
        for w in warnings:
            print(f"   - {w}")
    print("=" * width)


# -----------------------------------------------------------------------------
# Batch diagnostic
# -----------------------------------------------------------------------------

async def run_batch_diagnostic(args):
    client = WebsocketRpcClient(args.ws_url)
    seen = set()
    duplicates = 0
    total = 0
    all_hashes = []

    print(f"Connecting to {args.ws_url} ...")
    for i in range(args.n_requests):
        try:
            msg = await client.get_genome_training_batch(
                args.genome_merkle,
                args.model_id,
                args.batch_size,
                request_id=i + 1,
            )
        except Exception as e:
            print(f"[ERROR] Request {i + 1} failed: {e}")
            continue

        if msg.get("type") == "Error":
            print(f"[ERROR] Request {i + 1}: {msg.get('message')}")
            continue

        sequences = msg.get("sequences", [])
        print(f"\nRequest #{i + 1}: {len(sequences)} sequences")
        for seq in sequences:
            seq_hash = hashlib.md5(seq.encode()).hexdigest()[:16]
            gc = (seq.count("G") + seq.count("C")) / len(seq) * 100 if seq else 0.0
            print(f"  [{seq_hash}] len={len(seq)} GC={gc:.1f}%  {seq[:40]}...")
            if seq_hash in seen:
                duplicates += 1
            seen.add(seq_hash)
            all_hashes.append(seq_hash)
        total += 1

    print("\n" + "=" * 70)
    print(" BATCH VARIATION DIAGNOSTIC")
    print("=" * 70)
    print(f" Requests: {total}")
    print(f" Unique sequences: {len(seen)}")
    print(f" Duplicates: {duplicates}")
    if total:
        unique_ratio = len(seen) / (total * args.batch_size)
        print(f" Diversity ratio: {unique_ratio:.1%}")
    if duplicates == 0 and total > 0:
        print(" Status: all sequences are unique.")
    else:
        print(" Status: found repeated sequences - batch RNG may not be varying.")
    print("=" * 70)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def resolve_genome_source(args) -> str:
    if args.genome:
        return args.genome

    # Try the seed-node models directory, then the miner cache.
    candidates = [
        Path.home() / ".xenom-miner" / "genomes" / "grch38.xenom",
        Path.home() / ".xenom-miner" / "models" / "genomes" / "grch38.xenom",
        Path("grch38.xenom"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    if args.allow_download:
        target = Path("grch38.xenom")
        print(f"Downloading genome archive from {GENOME_DOWNLOAD_URL} ...")
        import requests
        with requests.get(GENOME_DOWNLOAD_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(target, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        return str(target)

    raise SystemExit(
        "No genome source provided and grch38.xenom not found.\n"
        "Options:\n"
        "  --genome <path.xenom|path.fasta>\n"
        "  --allow-download (fetches the canonical GRCh38 archive, ~700 MB)\n"
        "  place grch38.xenom in the working directory"
    )


def run_evaluation(args):
    genome_source = resolve_genome_source(args)

    with GrpcInferenceClient(args.grpc_addr) as grpc_client:
        # Verify the model is known to the seed-node.
        try:
            info = grpc_client.get_model_info(args.model_id)
            print(f"Model info: {args.model_id} active={info.active} "
                  f"version={info.version} hash={info.model_hash.hex()[:16]}...")
        except grpc.RpcError as e:
            print(f"[WARN] Could not get model info: {e}")

        evaluator = Mgm1Evaluator(
            grpc_client,
            model_id=args.model_id,
            seq_len=args.seq_len,
            mask_ratio=args.mask_ratio,
        )

        print(f"Building test set from {genome_source} ...")
        test_set = evaluator.build_test_set(
            genome_source, n=args.n_sequences, seed=args.test_seed
        )
        print(f"Extracted {len(test_set)} {args.seq_len}-bp sequences.")

        print("Running MLM evaluation (this may take a while) ...")
        metrics = evaluator.evaluate(test_set, test_seed=args.test_seed)

    history = load_history(args.history)
    previous = history[-1]["metrics"] if history else None
    trend = compute_trend(metrics, previous)
    warnings = validate_metrics(metrics, warn_threshold=args.warn_threshold)

    print_report(metrics, args.block, args.model_id, trend, warnings)

    record = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "block": args.block,
        "model_id": args.model_id,
        "metrics": metrics,
        "trend": trend,
    }
    history.append(record)
    save_history(args.history, history)
    print(f"Saved history to {args.history}")

    if warnings:
        return 2 if args.strict else 0
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate xeno/mgm-1 training progress on real GRCh38 DNA."
    )
    parser.add_argument(
        "--block",
        type=int,
        default=1,
        help="Block number to tag this evaluation (default: 1).",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Model id to evaluate (default: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--genome",
        help="Path to a .xenom archive or .fasta file.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Download the canonical grch38.xenom if not found locally.",
    )
    parser.add_argument(
        "--grpc-addr",
        default=DEFAULT_GRPC_ADDR,
        help=f"gRPC inference server address (default: {DEFAULT_GRPC_ADDR}).",
    )
    parser.add_argument(
        "--n-sequences",
        type=int,
        default=DEFAULT_N_SEQUENCES,
        help=f"Number of test sequences (default: {DEFAULT_N_SEQUENCES}).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help=f"Length of each test sequence in bases (default: {DEFAULT_SEQ_LEN}).",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=DEFAULT_MASK_RATIO,
        help=f"Fraction of bases to mask (default: {DEFAULT_MASK_RATIO}).",
    )
    parser.add_argument(
        "--test-seed",
        type=int,
        default=42,
        help="Random seed used to sample the test set and mask positions.",
    )
    parser.add_argument(
        "--history",
        default="mgm1_eval_history.json",
        help="JSON file to store/append evaluation history.",
    )
    parser.add_argument(
        "--warn-threshold",
        type=float,
        default=0.25,
        help="Accuracy below this value triggers a warning (default: 0.25).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status when warnings are present.",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Run the WebSocket batch-variation diagnostic instead of evaluation.",
    )
    parser.add_argument(
        "--ws-url",
        default=DEFAULT_WS_URL,
        help=f"WebSocket seed-node URL for diagnostics (default: {DEFAULT_WS_URL}).",
    )
    parser.add_argument(
        "--genome-merkle",
        default=DEFAULT_GENOME_MERKLE,
        help="Merkle root of the genome archive for diagnostics.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of sequences per batch in diagnostic mode.",
    )
    parser.add_argument(
        "--n-requests",
        type=int,
        default=10,
        help="Number of batches to request in diagnostic mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.diagnostic:
        try:
            asyncio.run(run_batch_diagnostic(args))
        except Exception as e:
            print(f"[FATAL] Diagnostic failed: {e}")
            return 1
        return 0

    return run_evaluation(args)


if __name__ == "__main__":
    sys.exit(main())
