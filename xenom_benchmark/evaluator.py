"""End-to-end benchmark evaluator: from genome windows to a BenchmarkRecord."""

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dataset import (
    GenomeAnnotation,
    SequenceWindow,
    load_annotation,
    load_windows,
    resolve_genome_source,
)
from .grpc_backend import Backend, ModelInfo, PredictionResult
from .masking import MaskingStrategy, mask_sequences
from .metrics import MaskedSample, MetricsComputer, MetricsResult
from .report import BenchmarkRecord
from .tokenizer import BenchmarkTokenizer
from .utils import BASES, BASE_SET, DEFAULT_SEQ_LEN, logger, now_iso, sha256_file, validate_bases


@dataclass
class EvaluatorConfig:
    """Configuration for the benchmark evaluator."""

    model_id: str = "xeno/mgm-1"
    block: int = 0
    genome: Optional[Path] = None
    seed: int = 42
    num_samples: int = 88
    seq_length: int = DEFAULT_SEQ_LEN
    mask_prob: float = 0.15
    batch_size: int = 1
    grpc_address: str = ""
    annotation: Optional[Path] = None
    vocab_size: Optional[int] = None
    timeout: float = 120.0
    tokenizer_path: Optional[Path] = None
    use_token_level: bool = True
    tags: List[str] = field(default_factory=list)


class GenomicTagger:
    """Assign genomic-region tags to masked positions."""

    def __init__(self, annotation: Optional[GenomeAnnotation] = None) -> None:
        self.annotation = annotation

    def tag_position(
        self, window: str, fragment_start: int, position: int, fragment_idx: int = 0, fragment_size: int = 0
    ) -> List[str]:
        """Return a list of region tags for a position."""
        tags: List[str] = []

        if self.annotation:
            abs_pos = fragment_idx * fragment_size + fragment_start + position
            for region_name in ["promoter", "enhancer", "exon", "intron", "cpg_island"]:
                if self.annotation.contains(region_name, abs_pos):
                    tags.append(region_name)
            if not tags:
                tags.append("intergenic")

        # GC and AT content are always computable from the local window.
        gc = (window.count("G") + window.count("C")) / len(window) if window else 0.0
        at = (window.count("A") + window.count("T")) / len(window) if window else 0.0
        if gc >= 0.6:
            tags.append("gc_rich")
        if at >= 0.6:
            tags.append("at_rich")

        return tags


def _align_prediction(
    original: str, predicted: str, original_pos: int
) -> str:
    """Return the predicted character at ``original_pos`` using diff alignment.

    For equal-length outputs the character at the same position is returned
    directly. For variable-length (e.g. BPE) outputs, a ``difflib`` alignment is
    used and the index is clamped to stay within a matched predicted block.
    """
    if len(predicted) == len(original) and 0 <= original_pos < len(predicted):
        return predicted[original_pos]

    matcher = difflib.SequenceMatcher(None, original, predicted)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if i1 <= original_pos < i2:
            if tag in ("equal", "replace"):
                block_len = j2 - j1
                offset = original_pos - i1
                if block_len > 0:
                    idx = j1 + min(offset, block_len - 1)
                    return predicted[idx]
            if tag == "delete":
                return "?"
    return "?"


def _extract_masked_predictions(
    sequence_id: int,
    window: SequenceWindow,
    mask_result,
    prediction: PredictionResult,
    tagger: GenomicTagger,
) -> List[MaskedSample]:
    """Build ``MaskedSample`` instances for every masked position."""
    original = mask_result.original
    predicted = prediction.predicted_sequence

    if mask_result.token_level and mask_result.tokens is not None:
        return _extract_token_level_samples(
            sequence_id, window, mask_result, prediction, tagger
        )

    # Legacy character-level (base reconstruction) path.
    samples: List[MaskedSample] = []
    for pos in mask_result.positions:
        true_base = original[pos]
        pred_base = _align_prediction(original, predicted, pos)

        if pred_base not in BASE_SET:
            logger.warning(
                "sequence %d position %d: predicted base %r is not a DNA base; "
                "treating as incorrect",
                sequence_id,
                pos,
                pred_base,
            )
            pred_base = next(b for b in BASES if b != true_base)

        region_tags = tagger.tag_position(
            original,
            window.start,
            pos,
            window.fragment_idx,
            window.fragment_size,
        )

        confidence = prediction.confidence
        if not np.isfinite(confidence):
            confidence = 0.0

        logits = None
        logits_labels = None
        if (
            prediction.logits_available
            and prediction.logits is not None
            and pos < prediction.logits.shape[0]
        ):
            logits = prediction.logits[pos]
            logits_labels = prediction.logits_labels

        samples.append(
            MaskedSample(
                sequence_id=sequence_id,
                position=pos,
                true_base=true_base,
                predicted_base=pred_base,
                confidence=confidence,
                logits=logits,
                logits_labels=logits_labels,
                region_tags=region_tags,
            )
        )

    return samples


def _extract_token_level_samples(
    sequence_id: int,
    window: SequenceWindow,
    mask_result,
    prediction: PredictionResult,
    tagger: GenomicTagger,
) -> List[MaskedSample]:
    """Build token-level ``MaskedSample`` instances using returned logits."""
    if mask_result.tokens is None or mask_result.token_ids is None:
        raise ValueError("token-level masking requires token strings and ids")

    tokens = mask_result.tokens
    token_ids = mask_result.token_ids
    masked_token_positions = mask_result.masked_token_positions or []
    masked_token_ids = mask_result.masked_token_ids or []

    # Build a map from token index to its string.
    token_id_to_str = {i: tokens[i] for i in range(len(tokens))}

    # Server returns the token indices where <mask> appeared in its tokenization.
    server_mask_positions = prediction.masked_positions
    server_logits = prediction.logits

    samples: List[MaskedSample] = []
    for local_idx, true_pos in enumerate(masked_token_positions):
        true_token_id = masked_token_ids[local_idx]

        # Find the corresponding row in the server logits.  In the common case
        # the server tokenization is identical to ours, so server_mask_positions
        # is the same list as masked_token_positions.
        logit_row = None
        if server_logits is not None and server_mask_positions is not None:
            try:
                server_idx = server_mask_positions.index(true_pos)
                if 0 <= server_idx < server_logits.shape[0]:
                    logit_row = server_logits[server_idx]
            except ValueError:
                pass

        # Predicted token is the argmax of the logit row.  When logits are
        # unavailable, fall back to aligning the predicted string at the
        # character position of the token.
        if logit_row is not None:
            predicted_token_id = int(np.argmax(logit_row))
        else:
            char_pos = sum(len(tokens[i]) for i in range(true_pos))
            predicted_token_id = -1
            if char_pos < len(prediction.predicted_sequence):
                predicted_token_str = prediction.predicted_sequence[char_pos]
                predicted_token_id = tokens.index(predicted_token_str) if predicted_token_str in tokens else -1

        true_token_str = token_id_to_str.get(true_token_id, "")
        predicted_token_str = token_id_to_str.get(predicted_token_id, "")

        # Character position for region tagging is the token start.
        char_pos = sum(len(tokens[i]) for i in range(true_pos))
        region_tags = tagger.tag_position(
            mask_result.original,
            window.start,
            char_pos,
            window.fragment_idx,
            window.fragment_size,
        )

        confidence = prediction.confidence
        if not np.isfinite(confidence):
            confidence = 0.0

        samples.append(
            MaskedSample(
                sequence_id=sequence_id,
                position=char_pos,
                true_base=true_token_str,
                predicted_base=predicted_token_str,
                confidence=confidence,
                logits=logit_row,
                logits_labels=tokens if logit_row is not None else None,
                region_tags=region_tags,
                token_level=True,
                true_token_id=true_token_id,
                predicted_token_id=predicted_token_id if predicted_token_id >= 0 else None,
            )
        )

    return samples


class Evaluator:
    """Coordinate dataset loading, inference and metric computation."""

    def __init__(
        self,
        backend: Backend,
        config: EvaluatorConfig,
    ) -> None:
        self.backend = backend
        self.config = config
        self.tokenizer = self._load_tokenizer()

    def _load_tokenizer(self) -> Optional[BenchmarkTokenizer]:
        """Load a tokenizer for token-level MLM evaluation if possible."""
        if not self.config.use_token_level:
            return None

        lower = self.config.model_id.lower()
        if "mgm" in lower or "mini-genome" in lower:
            return None

        # BPE/k-mer models such as DNABERT-2 and nucleotide-transformer must use
        # token-level masking; falling back to character-level would silently turn
        # the MLM benchmark into base reconstruction.
        requires_tokenizer = "dnabert" in lower or "nucleotide" in lower

        if self.config.tokenizer_path:
            try:
                return BenchmarkTokenizer.from_model_dir(self.config.tokenizer_path)
            except Exception as exc:
                if requires_tokenizer:
                    raise RuntimeError(
                        f"{self.config.model_id} requires a tokenizer, failed to load from {self.config.tokenizer_path}: {exc}"
                    ) from exc
                logger.warning("failed to load tokenizer from %s: %s", self.config.tokenizer_path, exc)

        try:
            return BenchmarkTokenizer.from_huggingface(self.config.model_id)
        except Exception as exc:
            if requires_tokenizer:
                raise RuntimeError(
                    f"{self.config.model_id} requires its Hugging Face tokenizer, download failed: {exc}"
                ) from exc
            logger.warning(
                "failed to download tokenizer for %s: %s; "
                "falling back to character-level masking",
                self.config.model_id,
                exc,
            )
        return None

    def _load_windows(self) -> List[SequenceWindow]:
        """Load the requested number of DNA windows from the genome source."""
        genome_path = resolve_genome_source(self.config.genome)
        return load_windows(
            genome_path,
            self.config.num_samples,
            self.config.seq_length,
            self.config.seed,
        )

    def _call_backend(self, masked: str) -> PredictionResult:
        """Invoke the inference backend with timeout handling."""
        return self.backend.predict(self.config.model_id, masked)

    def _collect_samples(
        self,
        windows: List[SequenceWindow],
    ) -> Tuple[List[MaskedSample], List[str]]:
        """Run inference on every masked window and collect samples."""
        strategy = MaskingStrategy.for_model(
            self.config.model_id,
            self.config.mask_prob,
            tokenizer=self.tokenizer,
        )
        tagger = GenomicTagger(
            load_annotation(self.config.annotation) if self.config.annotation else None
        )

        sequences = [w.sequence for w in windows]
        masked_results = mask_sequences(sequences, strategy, self.config.seed)

        all_samples: List[MaskedSample] = []
        warnings: List[str] = []

        for i, (window, mr) in enumerate(zip(windows, masked_results)):
            try:
                prediction = self._call_backend(mr.masked)
                warnings.extend(prediction.warnings)
                samples = _extract_masked_predictions(i, window, mr, prediction, tagger)
                all_samples.extend(samples)
            except Exception as exc:
                logger.error("sequence %d failed: %s", i, exc)
                warnings.append(f"sequence {i} failed: {exc}")

        return all_samples, warnings

    def _region_metrics(
        self, samples: List[MaskedSample], metrics_computer: MetricsComputer
    ) -> Dict[str, Dict[str, Any]]:
        """Compute per-region metrics for all observed tags."""
        region_names: set = set()
        for s in samples:
            region_names.update(s.region_tags)

        region_metrics: Dict[str, Dict[str, Any]] = {}
        for region in sorted(region_names):
            try:
                result = metrics_computer.compute_region(samples, region)
                region_metrics[region] = result
            except ValueError:
                continue
        return region_metrics

    def run(self) -> BenchmarkRecord:
        """Execute the full benchmark pipeline and return a record."""
        logger.info("Loading %d windows of length %d", self.config.num_samples, self.config.seq_length)
        windows = self._load_windows()
        genome_path = resolve_genome_source(self.config.genome)
        genome_hash = sha256_file(genome_path)

        logger.info("Requesting model info for %s", self.config.model_id)
        try:
            model_info = self.backend.get_model_info(self.config.model_id)
        except Exception as exc:
            logger.warning("could not get model info: %s", exc)
            model_info = None

        logger.info("Running inference on %d masked sequences", len(windows))
        samples, warnings = self._collect_samples(windows)
        if not samples:
            raise RuntimeError("no valid masked predictions were collected")

        token_level = any(s.token_level for s in samples)
        metrics_computer = MetricsComputer(
            vocab_size=self.config.vocab_size, token_level=token_level
        )
        metrics = metrics_computer.compute(samples)
        warnings.extend(metrics.warnings)

        region_metrics = self._region_metrics(samples, metrics_computer)

        record = BenchmarkRecord(
            block=self.config.block,
            model_id=self.config.model_id,
            model_version=model_info.version if model_info else "",
            model_hash=model_info.model_hash if model_info else None,
            genome_source=str(genome_path),
            genome_hash=genome_hash,
            seed=self.config.seed,
            num_samples=self.config.num_samples,
            seq_length=self.config.seq_length,
            mask_prob=self.config.mask_prob,
            grpc_address=self.config.grpc_address,
            vocab_size=self.config.vocab_size or metrics.vocab_size,
            metrics=metrics,
            region_metrics=region_metrics,
            warnings=warnings,
        )

        # Ensure the metrics dict is JSON-safe.
        record.metrics = _metrics_to_dict(metrics)
        record.region_metrics = {
            k: _metrics_to_dict(v) for k, v in region_metrics.items()
        }

        return record


def _metrics_to_dict(metrics: MetricsResult) -> Dict[str, Any]:
    """Serialize a MetricsResult to a JSON-safe dictionary."""
    from dataclasses import asdict
    from .report import _json_friendly
    return _json_friendly(asdict(metrics))
