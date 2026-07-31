"""Blockchain inference backends. The default implementation uses the existing
``xenom.inference.Inference`` gRPC service and detects whether the response
includes full logits or only a predicted sequence + confidence.
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .utils import (
    BASE_SET,
    DEFAULT_GRPC_ADDR,
    DEFAULT_TIMEOUT,
    add_scripts_to_path,
    logger,
    validate_confidence,
)

add_scripts_to_path()

try:
    import grpc
    import inference_pb2
    import inference_pb2_grpc
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing Python dependencies or generated gRPC stubs.\n"
        "Install: pip install grpcio protobuf requests\n"
        "Generate stubs: python -m grpc_tools.protoc "
        "--python_out=scripts/xenom_eval --grpc_python_out=scripts/xenom_eval "
        "-Iproto proto/inference.proto"
    ) from exc


@dataclass
class ModelInfo:
    """Lightweight model metadata returned by the backend."""

    model_id: str
    name: str
    version: str
    category: str
    active: bool
    verified: bool
    model_hash: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Result of a single prediction call to a model backend."""

    predicted_sequence: str
    confidence: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    model_version: str = ""
    logits: Optional[np.ndarray] = None  # shape [num_masks, vocab_size]
    logits_available: bool = False
    logits_labels: Optional[List[str]] = None  # string label for each logit column
    top_k_indices: Optional[np.ndarray] = None  # shape [num_masks, k]
    top_k_probs: Optional[np.ndarray] = None  # shape [num_masks, k]
    masked_positions: Optional[List[int]] = None  # token indices of each mask
    vocab_size: Optional[int] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class Backend(ABC):
    """Abstract inference backend."""

    @abstractmethod
    def predict(self, model_id: str, masked_sequence: str) -> PredictionResult:
        """Return the prediction for a single masked DNA sequence."""
        ...

    @abstractmethod
    def get_model_info(self, model_id: str) -> ModelInfo:
        """Return metadata for `model_id`."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the backend and release resources."""
        ...


class GrpcBackend(Backend):
    """gRPC backend using the Xenomorph ``xenom.inference.Inference`` service."""

    def __init__(
        self,
        address: str = DEFAULT_GRPC_ADDR,
        timeout: float = DEFAULT_TIMEOUT,
        block_height: int = 0,
    ):
        self.address = address
        self.timeout = timeout
        self.block_height = block_height
        self.channel = grpc.insecure_channel(address)
        self.stub = inference_pb2_grpc.InferenceStub(self.channel)

    def set_block_height(self, block_height: int) -> None:
        """Set the checkpoint block height for subsequent inference calls."""
        self.block_height = block_height

    def _call_evaluate_masked_llm(self, model_id: str, masked_sequence: str) -> inference_pb2.EvaluateMaskedLlmResponse:
        """Make the gRPC EvaluateMaskedLlm call with a deadline."""
        query_id = f"xenom-bench-{int(time.time() * 1000)}"
        request = inference_pb2.EvaluateMaskedLlmRequest(
            model_id=model_id,
            input_data=masked_sequence.encode("utf-8"),
            query_id=query_id,
            block_height=self.block_height,
        )
        try:
            return self.stub.EvaluateMaskedLlm(request, timeout=self.timeout)
        except grpc.RpcError as exc:
            code = exc.code() if hasattr(exc, "code") else "UNKNOWN"
            details = exc.details() if hasattr(exc, "details") else str(exc)
            raise ConnectionError(f"gRPC EvaluateMaskedLlm failed [{code}]: {details}") from exc

    def _call_predict(self, model_id: str, masked_sequence: str) -> inference_pb2.PredictResponse:
        """Make the gRPC Predict call with a deadline."""
        query_id = f"xenom-bench-{int(time.time() * 1000)}"
        request = inference_pb2.PredictRequest(
            model_id=model_id,
            input_data=masked_sequence.encode("utf-8"),
            query_id=query_id,
            block_height=self.block_height,
        )
        try:
            return self.stub.Predict(request, timeout=self.timeout)
        except grpc.RpcError as exc:
            code = exc.code() if hasattr(exc, "code") else "UNKNOWN"
            details = exc.details() if hasattr(exc, "details") else str(exc)
            raise ConnectionError(f"gRPC Predict failed [{code}]: {details}") from exc

    def _parse_output_data(self, raw: bytes) -> Tuple[str, Optional[np.ndarray], Dict[str, Any]]:
        """Parse `output_data` from a PredictResponse.

        The current protocol encodes the predicted DNA sequence as raw bytes.
        Future versions may wrap the output in a JSON object that also contains
        ``logits`` or ``top_k``. This parser tries the JSON form first and falls
        back to a plain string.
        """
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")

        # Try to parse as a JSON envelope.
        if text.strip().startswith("{"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                sequence = payload.get("sequence", payload.get("predicted", ""))
                logits = self._extract_logits(payload)
                return str(sequence), logits, payload

        return text, None, {}

    def _extract_logits(self, payload: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract and validate a logits tensor from a JSON payload."""
        logits_raw = payload.get("logits")
        if logits_raw is None:
            return None
        try:
            arr = np.asarray(logits_raw, dtype=np.float64)
        except (ValueError, TypeError):
            logger.warning("logits field present but not a numeric array; ignoring")
            return None

        if arr.ndim not in (1, 2):
            logger.warning("logits must be 1-D or 2-D; ignoring")
            return None

        if not np.isfinite(arr).all():
            logger.warning("logits contain NaN or Inf; ignoring")
            return None

        # If a single position, make it [1, vocab_size].
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        return arr

    def _extract_top_k(
        self, payload: Dict[str, Any]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Extract optional top-k indices/probabilities from a JSON payload."""
        indices = payload.get("top_k_indices")
        probs = payload.get("top_k_probs")
        if indices is None or probs is None:
            return None
        try:
            idx = np.asarray(indices, dtype=np.int64)
            pr = np.asarray(probs, dtype=np.float64)
        except (ValueError, TypeError):
            return None
        if not np.isfinite(pr).all():
            return None
        if idx.ndim == 1:
            idx = idx[np.newaxis, :]
        if pr.ndim == 1:
            pr = pr[np.newaxis, :]
        return idx, pr

    def predict(self, model_id: str, masked_sequence: str) -> PredictionResult:
        """Call the gRPC ``EvaluateMaskedLlm`` endpoint when available.

        If the new endpoint is unavailable, fall back to the legacy ``Predict``
        endpoint, which does not expose full logits.
        """
        try:
            return self._predict_evaluate_masked_llm(model_id, masked_sequence)
        except ConnectionError as exc:
            if "UNIMPLEMENTED" in str(exc).upper():
                logger.warning("EvaluateMaskedLlm not implemented on server; falling back to Predict: %s", exc)
                return self._predict_legacy(model_id, masked_sequence)
            raise

    def _predict_evaluate_masked_llm(self, model_id: str, masked_sequence: str) -> PredictionResult:
        """Call the gRPC ``EvaluateMaskedLlm`` endpoint and validate the response."""
        response = self._call_evaluate_masked_llm(model_id, masked_sequence)

        if not response.output_data:
            raise ValueError("EvaluateMaskedLlmResponse.output_data is empty")

        predicted = response.output_data.decode("utf-8", errors="replace")
        validate_confidence(response.confidence)

        logits: Optional[np.ndarray] = None
        if response.masked_logits and response.masked_positions and response.logits_vocab_size:
            n_masks = len(response.masked_positions)
            vocab_size = response.logits_vocab_size
            try:
                logits = np.asarray(response.masked_logits, dtype=np.float64).reshape(n_masks, vocab_size)
            except ValueError:
                logger.warning("masked_logits length does not match num_masks * vocab_size; ignoring logits")
                logits = None

        result = PredictionResult(
            predicted_sequence=predicted,
            confidence=response.confidence,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=response.latency_ms,
            model_version=response.model_version,
            logits=logits,
            logits_available=logits is not None,
            masked_positions=list(response.masked_positions),
            vocab_size=response.logits_vocab_size,
            metadata={},
        )

        invalid = set(predicted) - BASE_SET
        if invalid:
            result.warnings.append(
                f"predicted sequence contains non-DNA characters: {sorted(invalid)}"
            )

        if not predicted:
            raise ValueError("predicted sequence is empty after decoding")

        return result

    def _predict_legacy(self, model_id: str, masked_sequence: str) -> PredictionResult:
        """Call the gRPC ``Predict`` endpoint and validate the response."""
        response = self._call_predict(model_id, masked_sequence)

        if not response.output_data:
            raise ValueError("PredictResponse.output_data is empty")

        predicted, logits, payload = self._parse_output_data(response.output_data)
        validate_confidence(response.confidence)

        logits_labels = payload.get("logits_labels") if isinstance(payload, dict) else None
        top_k = self._extract_top_k(payload) if isinstance(payload, dict) else None

        result = PredictionResult(
            predicted_sequence=predicted,
            confidence=response.confidence,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=response.latency_ms,
            model_version=response.model_version,
            logits=logits,
            logits_available=logits is not None,
            logits_labels=logits_labels,
            top_k_indices=top_k[0] if top_k else None,
            top_k_probs=top_k[1] if top_k else None,
            metadata={},
        )

        invalid = set(predicted) - BASE_SET
        if invalid:
            result.warnings.append(
                f"predicted sequence contains non-DNA characters: {sorted(invalid)}"
            )

        if not predicted:
            raise ValueError("predicted sequence is empty after decoding")

        return result

    def get_model_info(self, model_id: str, block_height: int = 0) -> ModelInfo:
        """Call the gRPC ``GetModelInfo`` endpoint."""
        request = inference_pb2.ModelInfoRequest(
            model_id=model_id,
            block_height=block_height,
        )
        try:
            response = self.stub.GetModelInfo(request, timeout=self.timeout)
        except grpc.RpcError as exc:
            code = exc.code() if hasattr(exc, "code") else "UNKNOWN"
            details = exc.details() if hasattr(exc, "details") else str(exc)
            raise ConnectionError(f"gRPC GetModelInfo failed [{code}]: {details}") from exc

        return ModelInfo(
            model_id=response.model_id,
            name=response.name,
            version=response.version,
            category=response.category,
            active=response.active,
            verified=response.verified,
            model_hash=response.model_hash.hex() if response.model_hash else None,
            metadata=dict(response.metadata or {}),
        )

    def list_models(self) -> List[ModelInfo]:
        """Call the gRPC ``ListModels`` endpoint."""
        request = inference_pb2.ListModelsRequest(active_only=False)
        try:
            response = self.stub.ListModels(request, timeout=self.timeout)
        except grpc.RpcError as exc:
            raise ConnectionError(f"gRPC ListModels failed: {exc}") from exc

        return [
            ModelInfo(
                model_id=m.model_id,
                name=m.name,
                version=m.version,
                category=m.category,
                active=m.active,
                verified=m.verified,
            )
            for m in response.models
        ]

    def health_check(self) -> bool:
        """Return True if the gRPC health check reports healthy."""
        request = inference_pb2.HealthCheckRequest(service="inference")
        try:
            response = self.stub.HealthCheck(request, timeout=self.timeout)
            return response.healthy
        except grpc.RpcError as exc:
            logger.warning("gRPC HealthCheck failed: %s", exc)
            return False

    def close(self) -> None:
        """Close the gRPC channel."""
        self.channel.close()

    def __enter__(self) -> "GrpcBackend":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
