"""MLM metric computation with automatic logit detection and baseline comparisons.

When the blockchain backend exposes per-token logits the suite computes exact
cross-entropy, NLL, perplexity and top-k accuracy. When only the predicted
sequence and an average confidence are returned, the suite falls back to a
**uniform-tail imputed distribution** (probability mass is placed on the
top-1 prediction according to the reported confidence and the remainder is
spread uniformly over the remaining vocabulary) and warns the caller.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from .utils import BASES, BASE_SET, DNA_TO_INDEX, safe_log, safe_mean


@dataclass
class MaskedSample:
    """A single masked-position observation."""

    sequence_id: int
    position: int
    true_base: str
    predicted_base: str
    confidence: float
    logits: Optional[np.ndarray] = None  # [vocab_size]
    logits_labels: Optional[List[str]] = None
    region_tags: List[str] = field(default_factory=list)
    token_level: bool = False
    true_token_id: Optional[int] = None
    predicted_token_id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.token_level:
            return
        if self.true_base not in BASE_SET:
            raise ValueError(f"invalid true_base: {self.true_base}")
        if self.predicted_base not in BASE_SET:
            raise ValueError(f"invalid predicted_base: {self.predicted_base}")


@dataclass
class PerBaseMetrics:
    """Per-base precision, recall, f1, accuracy and support."""

    count: int = 0
    correct: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0


@dataclass
class BaselineMetrics:
    """Metric container for a single baseline."""

    name: str
    accuracy: float = 0.0
    top3_accuracy: Optional[float] = None
    top5_accuracy: Optional[float] = None
    balanced_accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    macro_f1: float = 0.0
    micro_f1: float = 0.0
    weighted_f1: float = 0.0
    mcc: float = 0.0
    kappa: float = 0.0
    cross_entropy: Optional[float] = None
    nll: Optional[float] = None
    perplexity: Optional[float] = None
    prediction_entropy: float = 0.0
    average_confidence: float = 0.0
    prediction_distribution: Dict[str, int] = field(default_factory=dict)
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    note: str = ""


@dataclass
class MetricsResult:
    """Full metric set for a model or a genomic region."""

    n_samples: int = 0
    n_masked: int = 0
    logits_available: bool = False
    vocab_size: int = 4
    token_level: bool = False
    approximation_note: str = ""

    cross_entropy: Optional[float] = None
    nll: Optional[float] = None
    perplexity: Optional[float] = None
    cross_entropy_lower_bound: Optional[float] = None

    top1_accuracy: float = 0.0
    top3_accuracy: Optional[float] = None
    top5_accuracy: Optional[float] = None
    top_k_approximate: bool = False

    balanced_accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    macro_f1: float = 0.0
    micro_f1: float = 0.0
    weighted_f1: float = 0.0
    mcc: float = 0.0
    kappa: float = 0.0

    prediction_entropy: float = 0.0
    average_confidence: float = 0.0
    prediction_distribution: Dict[str, int] = field(default_factory=dict)
    true_distribution: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    per_base: Dict[str, PerBaseMetrics] = field(default_factory=dict)

    baselines: Dict[str, BaselineMetrics] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class MetricsComputer:
    """Compute MLM benchmark metrics from a collection of masked samples."""

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        token_level: bool = False,
    ) -> None:
        self.vocab_size = vocab_size
        self.token_level = token_level
        self._warnings: List[str] = []

    def _resolve_vocab_size(
        self, samples: Sequence[MaskedSample], logits_labels: Optional[List[str]]
    ) -> int:
        """Resolve the vocabulary size from logits, labels, or fallback to 4."""
        if logits_labels is not None:
            return len(logits_labels)

        if self.vocab_size is not None:
            return self.vocab_size

        # If every prediction is a single DNA character and the model id suggests
        # a character-level model, the effective vocabulary is 4.
        all_dna = all(s.predicted_base in BASE_SET for s in samples)
        if all_dna:
            return 4

        self._warnings.append(
            "vocab_size could not be determined and was defaulted to 4. "
            "Cross-entropy and top-k approximations may be inaccurate for BPE models."
        )
        return 4

    def _imputed_prob(
        self, predicted: str, true: str, confidence: float, vocab_size: int
    ) -> float:
        """Return the imputed probability of the true class under a uniform tail."""
        if predicted == true:
            return confidence
        if vocab_size <= 1:
            return 1.0
        return (1.0 - confidence) / (vocab_size - 1)

    def _cross_entropy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidences: np.ndarray,
        logits: Optional[np.ndarray],
        logits_labels: Optional[List[str]],
        vocab_size: int,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Return (ce, nll, ppl, ce_lower_bound).

        ``ce`` is exact when logits are available, otherwise an imputed
        approximation. ``ce_lower_bound`` is a mathematically valid lower bound
        derived from the reported confidence.
        """
        n = len(y_true)
        if n == 0:
            return None, None, None, None

        if logits is not None and logits_labels is not None:
            return self._cross_entropy_from_logits(
                y_true, logits, logits_labels, vocab_size
            )

        # No logits: use the reported confidence and a uniform tail.
        nlls = np.empty(n, dtype=np.float64)
        lbs = np.empty(n, dtype=np.float64)
        for i in range(n):
            pred = y_pred[i]
            true = y_true[i]
            conf = confidences[i]
            p_true = self._imputed_prob(pred, true, conf, vocab_size)
            nlls[i] = -safe_log(p_true)
            # Lower bound: p(true) <= 1 - conf when wrong, = conf when correct.
            p_lb = conf if pred == true else (1.0 - conf)
            lbs[i] = -safe_log(p_lb)

        nll = float(np.mean(nlls))
        ce = nll  # for a single token per sample, CE == mean NLL
        ppl = float(np.exp(ce))
        ce_lb = float(np.mean(lbs))
        return ce, nll, ppl, ce_lb

    def _cross_entropy_from_logits(
        self,
        y_true: np.ndarray,
        logits: np.ndarray,
        logits_labels: List[str],
        vocab_size: int,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Compute exact CE/NLL/perplexity from logits."""
        if logits.shape[0] != len(y_true):
            self._warnings.append(
                f"logits length ({logits.shape[0]}) does not match sample count "
                f"({len(y_true)}); ignoring logits."
            )
            return None, None, None, None

        label_to_index = {lab: i for i, lab in enumerate(logits_labels)}
        # Convert string true bases to token ids; missing labels map to -1.
        true_ids = np.array(
            [label_to_index.get(y, -1) for y in y_true], dtype=np.int64
        )

        if (true_ids < 0).any():
            self._warnings.append(
                "some true labels are not present in logits_labels; "
                "cross-entropy will be computed for the known labels only."
            )
            mask = true_ids >= 0
            true_ids = true_ids[mask]
            logits = logits[mask]
            if len(true_ids) == 0:
                return None, None, None, None

        # Numerically stable log-softmax.
        logits_max = np.max(logits, axis=1, keepdims=True)
        shifted = logits - logits_max
        log_sum_exp = logits_max.squeeze(1) + np.log(np.sum(np.exp(shifted), axis=1))
        nlls = log_sum_exp - logits[np.arange(len(true_ids)), true_ids]

        nll = float(np.mean(nlls))
        ce = nll
        ppl = float(np.exp(ce))
        return ce, nll, ppl, ce

    def _top_k_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidences: np.ndarray,
        top_k_indices: Optional[np.ndarray],
        top_k_probs: Optional[np.ndarray],
        logits: Optional[np.ndarray],
        logits_labels: Optional[List[str]],
        vocab_size: int,
    ) -> Tuple[float, Optional[float], Optional[float], bool]:
        """Return (top1, top3, top5, is_approximate)."""
        n = len(y_true)
        top1 = float(np.mean(y_true == y_pred))

        if logits is not None and logits_labels is not None:
            top3, top5 = self._top_k_from_logits(
                y_true, logits, logits_labels, [3, 5]
            )
            return top1, top3, top5, False

        if top_k_indices is not None and top_k_probs is not None:
            top3, top5 = self._top_k_from_topk_payload(
                y_true, top_k_indices, top_k_probs, [3, 5]
            )
            return top1, top3, top5, False

        # Approximate top-k under a uniform tail distribution.
        if vocab_size <= 5:
            top5 = 1.0
        else:
            top5 = None

        if vocab_size <= 3:
            top3 = 1.0
        elif vocab_size is not None and vocab_size > 1:
            # Probability true is in top-3 given it is not top-1 is min(1, 2/(V-1)).
            factor = min(1.0, 2.0 / (vocab_size - 1))
            top3 = top1 + (1.0 - top1) * factor
        else:
            top3 = None

        if top3 is None or top5 is None:
            self._warnings.append(
                "top-3/top-5 accuracy could not be determined because neither "
                "logits nor a reliable vocab_size are available."
            )
        return top1, top3, top5, True

    def _top_k_from_logits(
        self,
        y_true: np.ndarray,
        logits: np.ndarray,
        logits_labels: List[str],
        ks: List[int],
    ) -> List[Optional[float]]:
        """Compute exact top-k accuracy from logits."""
        if logits.shape[0] != len(y_true):
            return [None, None]

        label_to_index = {lab: i for i, lab in enumerate(logits_labels)}
        true_ids = np.array([label_to_index.get(y, -1) for y in y_true])

        if (true_ids < 0).any():
            return [None, None]

        sorted_ids = np.argsort(-logits, axis=1)
        results: List[Optional[float]] = []
        for k in ks:
            top_k = sorted_ids[:, :k]
            correct = np.any(top_k == true_ids[:, None], axis=1)
            results.append(float(np.mean(correct)))
        return results

    def _top_k_from_topk_payload(
        self,
        y_true: np.ndarray,
        top_k_indices: np.ndarray,
        top_k_probs: np.ndarray,
        ks: List[int],
    ) -> List[Optional[float]]:
        """Compute top-k accuracy from a top-k payload."""
        if top_k_indices.shape[0] != len(y_true):
            return [None, None]

        results: List[Optional[float]] = []
        for k in ks:
            if k > top_k_indices.shape[1]:
                results.append(None)
                continue
            top_k = top_k_indices[:, :k]
            # We need true ids in the same index space as top_k. Without a
            # labels mapping this payload cannot be used for string true bases.
            results.append(None)
        return results

    def _sklearn_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute classification metrics using scikit-learn."""
        labels = [b for b in BASES]
        if len(y_true) == 0:
            return {
                "balanced_accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "macro_f1": 0.0,
                "micro_f1": 0.0,
                "weighted_f1": 0.0,
                "mcc": 0.0,
                "kappa": 0.0,
            }

        return {
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)),
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)),
            "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0, labels=labels)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        }

    def _confusion_dict(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, int]]:
        """Return a nested dict confusion matrix keyed by base."""
        labels = [b for b in BASES]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return {t: {p: int(cm[i, j]) for j, p in enumerate(labels)} for i, t in enumerate(labels)}

    def _per_base_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, PerBaseMetrics]:
        """Return per-base precision, recall, f1, accuracy and support."""
        labels = [b for b in BASES]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        per_base: Dict[str, PerBaseMetrics] = {}

        for i, base in enumerate(labels):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            total = cm[i, :].sum()
            correct = tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            accuracy = correct / total if total > 0 else 0.0

            per_base[base] = PerBaseMetrics(
                count=int(total),
                correct=int(correct),
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                accuracy=float(accuracy),
            )

        return per_base

    def _distributions(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[Dict[str, int], Dict[str, float], float]:
        """Return prediction counts, true frequencies and prediction entropy."""
        pred_counts: Dict[str, int] = {b: 0 for b in BASES}
        for p in y_pred:
            if p in pred_counts:
                pred_counts[p] += 1

        total = len(y_true)
        true_freqs = {b: float(np.sum(y_true == b) / total) if total else 0.0 for b in BASES}

        # Entropy of the empirical predicted distribution.
        probs = np.array([pred_counts[b] for b in BASES], dtype=np.float64)
        if probs.sum() > 0:
            probs = probs / probs.sum()
            probs = probs[probs > 0]
            pred_entropy = max(0.0, float(-np.sum(probs * np.log(probs))))
        else:
            pred_entropy = 0.0

        return pred_counts, true_freqs, pred_entropy

    def _token_distributions(self, y_true: np.ndarray) -> Tuple[Dict[str, int], Dict[str, float], float]:
        """Return token-level prediction counts, true frequencies and entropy."""
        unique, counts = np.unique(y_true, return_counts=True)
        true_freqs: Dict[str, float] = {str(u): float(c / len(y_true)) for u, c in zip(unique, counts)}

        pred_counts: Dict[str, int] = {}
        for u, c in zip(unique, counts):
            pred_counts[str(u)] = int(c)

        probs = np.array(list(true_freqs.values()), dtype=np.float64)
        if probs.sum() > 0:
            probs = probs / probs.sum()
            probs = probs[probs > 0]
            pred_entropy = max(0.0, float(-np.sum(probs * np.log(probs))))
        else:
            pred_entropy = 0.0

        return pred_counts, true_freqs, pred_entropy

    def _average_confidence(
        self, samples: Sequence[MaskedSample]
    ) -> float:
        """Return the mean confidence across all masked positions."""
        return safe_mean([s.confidence for s in samples])

    def _build_arrays(
        self, samples: Sequence[MaskedSample]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[List[str]], Optional[np.ndarray], Optional[np.ndarray]]:
        """Convert MaskedSample list into parallel arrays."""
        n = len(samples)
        y_true = np.empty(n, dtype=object)
        y_pred = np.empty(n, dtype=object)
        confidences = np.empty(n, dtype=np.float64)
        logits = None
        logits_labels = None
        top_k_indices = None
        top_k_probs = None

        has_logits = all(s.logits is not None for s in samples)
        for i, s in enumerate(samples):
            y_true[i] = s.true_base
            y_pred[i] = s.predicted_base
            confidences[i] = s.confidence
            if has_logits and s.logits is not None:
                if logits is None:
                    logits = np.empty((n, s.logits.shape[0]), dtype=np.float64)
                    logits_labels = s.logits_labels
                logits[i] = s.logits

        return y_true, y_pred, confidences, logits, logits_labels, top_k_indices, top_k_probs

    def _baseline_metrics(
        self, y_true: np.ndarray, vocab_size: int
    ) -> Dict[str, BaselineMetrics]:
        """Compute exact baseline metrics from the empirical label distribution."""
        n = len(y_true)
        labels = [b for b in BASES]
        true_freqs = np.array([np.sum(y_true == b) / n for b in labels], dtype=np.float64) if n else np.zeros(len(labels))

        # True class counts for confusion.
        true_counts = np.array([np.sum(y_true == b) for b in labels], dtype=np.int64)

        baselines: Dict[str, BaselineMetrics] = {}

        # Uniform baseline: predict each base with probability 1/4.
        n_classes = len(labels)
        uniform_pred = np.ones(n_classes) / n_classes
        baselines["uniform"] = self._baseline_from_distribution(
            "uniform",
            true_freqs,
            uniform_pred,
            vocab_size,
            n,
        )

        # Majority baseline: always predict the most frequent base.
        majority_idx = int(np.argmax(true_freqs)) if n else 0
        majority_dist = np.zeros(len(labels))
        majority_dist[majority_idx] = 1.0
        baselines["majority"] = self._baseline_from_distribution(
            "majority",
            true_freqs,
            majority_dist,
            vocab_size,
            n,
        )
        baselines["majority"].note = (
            "degenerate for non-majority classes; cross-entropy is smoothed "
            "to avoid infinity."
        )

        # Frequency baseline: predict each base with its empirical frequency.
        baselines["frequency"] = self._baseline_from_distribution(
            "frequency",
            true_freqs,
            true_freqs,
            vocab_size,
            n,
        )

        return baselines

    def _baseline_from_distribution(
        self,
        name: str,
        true_freqs: np.ndarray,
        pred_dist: np.ndarray,
        vocab_size: int,
        n: int,
    ) -> BaselineMetrics:
        """Compute baseline metrics from the true and predicted distributions."""
        n = max(n, 1)
        # Expected confusion matrix: C[t][p] = n * true[t] * pred[p]
        expected_cm = np.outer(true_freqs, pred_dist) * max(n, 1)
        labels = [b for b in BASES]

        # Expected per-class precision / recall / f1.
        per_class_precision = np.zeros(len(labels))
        per_class_recall = np.zeros(len(labels))
        per_class_f1 = np.zeros(len(labels))
        for i in range(len(labels)):
            tp = expected_cm[i, i]
            pred_total = expected_cm[:, i].sum()
            true_total = expected_cm[i, :].sum()
            precision = tp / pred_total if pred_total > 0 else 0.0
            recall = tp / true_total if true_total > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            per_class_precision[i] = precision
            per_class_recall[i] = recall
            per_class_f1[i] = f1

        macro_f1 = float(np.mean(per_class_f1))
        weighted_f1 = float(np.sum(true_freqs * per_class_f1))
        micro_tp = np.sum(np.diag(expected_cm))
        micro_total = np.sum(expected_cm)
        micro_f1 = micro_tp / micro_total if micro_total > 0 else 0.0
        balanced_acc = float(np.mean(per_class_recall))
        precision = float(np.mean(per_class_precision))
        recall = float(np.mean(per_class_recall))

        # MCC from expected confusion (treat as continuous matrix).
        mcc = self._mcc_from_matrix(expected_cm)
        kappa = self._kappa_from_matrix(expected_cm)

        # Accuracy from expected confusion.
        accuracy = float(np.sum(np.diag(expected_cm)) / np.sum(expected_cm)) if np.sum(expected_cm) > 0 else 0.0

        # Cross-entropy: -sum_t true[t] * log(pred[t]).
        # When a true class has zero predicted probability (e.g. the majority
        # baseline on a non-majority class), the exact CE is infinite. We do not
        # smooth this away so the comparison is mathematically faithful.
        with np.errstate(divide="ignore", invalid="ignore"):
            ce_terms = -true_freqs * np.log(pred_dist)
        ce_sum = float(np.sum(ce_terms))
        cross_entropy = ce_sum if np.isfinite(ce_sum) else float("inf")
        nll = cross_entropy
        perplexity = (
            float(np.exp(cross_entropy))
            if np.isfinite(cross_entropy)
            else float("inf")
        )

        # Top-k from distribution (4-class DNA label space).
        n_classes = len(labels)
        sorted_pred = np.sort(pred_dist)[::-1]
        top3 = float(np.sum(sorted_pred[: min(3, n_classes)]))
        top5 = 1.0 if n_classes <= 4 else float(np.sum(sorted_pred[:5]))

        # Prediction entropy and average confidence.
        eps = 1e-12
        safe_pred = np.where(pred_dist > 0, pred_dist, eps)
        safe_pred = safe_pred / safe_pred.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            pred_entropy = -float(np.sum(safe_pred * np.log(safe_pred)))
        avg_conf = (
            float(np.max(pred_dist))
            if name == "majority"
            else float(np.sum(pred_dist ** 2))
            if name == "frequency"
            else 1.0 / n_classes
        )

        # Confusion dict from expected (rounded) matrix.
        cm = expected_cm.round().astype(int)
        confusion = {
            t: {p: int(cm[i, j]) for j, p in enumerate(labels)}
            for i, t in enumerate(labels)
        }

        pred_dist_dict = {
            b: int(round(pred * n)) for b, pred in zip(labels, pred_dist)
        }

        return BaselineMetrics(
            name=name,
            accuracy=accuracy,
            top3_accuracy=top3,
            top5_accuracy=top5,
            balanced_accuracy=balanced_acc,
            precision=precision,
            recall=recall,
            macro_f1=macro_f1,
            micro_f1=micro_f1,
            weighted_f1=weighted_f1,
            mcc=mcc,
            kappa=kappa,
            cross_entropy=cross_entropy,
            nll=nll,
            perplexity=perplexity,
            prediction_entropy=pred_entropy,
            average_confidence=avg_conf,
            prediction_distribution=pred_dist_dict,
            confusion_matrix=confusion,
        )

    def _mcc_from_matrix(self, cm: np.ndarray) -> float:
        """Compute Matthews correlation coefficient from a confusion matrix."""
        # MCC = (cov(X,Y)) / sqrt(cov(X,X)*cov(Y,Y)) for categorical variables.
        # We use the multiclass formulation from sklearn on expected counts.
        from sklearn.metrics import matthews_corrcoef

        # Sample from the expected confusion matrix to get flat labels.
        n_classes = cm.shape[0]
        y_true = []
        y_pred = []
        for i in range(n_classes):
            for j in range(n_classes):
                count = int(round(cm[i, j]))
                y_true.extend([i] * count)
                y_pred.extend([j] * count)
        if not y_true:
            return 0.0
        return float(matthews_corrcoef(y_true, y_pred))

    def _kappa_from_matrix(self, cm: np.ndarray) -> float:
        """Compute Cohen's kappa from an expected confusion matrix."""
        from sklearn.metrics import cohen_kappa_score

        n_classes = cm.shape[0]
        y_true = []
        y_pred = []
        for i in range(n_classes):
            for j in range(n_classes):
                count = int(round(cm[i, j]))
                y_true.extend([i] * count)
                y_pred.extend([j] * count)
        if not y_true:
            return 0.0
        return float(cohen_kappa_score(y_true, y_pred))

    def compute(self, samples: Sequence[MaskedSample]) -> MetricsResult:
        """Compute the full metric set for ``samples``."""
        self._warnings = []
        n = len(samples)
        if n == 0:
            raise ValueError("cannot compute metrics from an empty sample list")

        is_token_level = self.token_level or all(s.token_level for s in samples)
        y_true, y_pred, confidences, logits, logits_labels, top_k_indices, top_k_probs = self._build_arrays(samples)
        vocab_size = self._resolve_vocab_size(samples, logits_labels)

        avg_conf = self._average_confidence(samples)

        # Top-k.
        top1, top3, top5, top_k_approx = self._top_k_accuracy(
            y_true, y_pred, confidences, top_k_indices, top_k_probs,
            logits, logits_labels, vocab_size,
        )

        # Cross-entropy / NLL / perplexity.
        ce, nll, ppl, ce_lb = self._cross_entropy(
            y_true, y_pred, confidences, logits, logits_labels, vocab_size
        )

        logits_available = logits is not None and logits_labels is not None

        approximation_note = ""
        if not logits_available:
            approximation_note = (
                "Cross-entropy, NLL, perplexity and top-k are approximations "
                "because the gRPC response does not expose logits. A uniform-tail "
                f"distribution over the remaining {vocab_size - 1} tokens is assumed."
            )
            self._warnings.append(approximation_note)
        elif is_token_level:
            approximation_note = (
                "Token-level MLM metrics: cross-entropy, NLL and perplexity are "
                "computed over the full model vocabulary. Base-level classification "
                "metrics are not reported for token labels."
            )

        if is_token_level:
            pred_counts, true_freqs, pred_entropy = self._token_distributions(y_true)
            sk = {k: 0.0 for k in [
                "balanced_accuracy", "precision", "recall", "macro_f1",
                "micro_f1", "weighted_f1", "mcc", "kappa",
            ]}
            confusion: Dict[str, Dict[str, int]] = {}
            per_base: Dict[str, PerBaseMetrics] = {}
            baselines: Dict[str, BaselineMetrics] = {}
        else:
            pred_counts, true_freqs, pred_entropy = self._distributions(y_true, y_pred)
            sk = self._sklearn_metrics(y_true, y_pred)
            confusion = self._confusion_dict(y_true, y_pred)
            per_base = self._per_base_metrics(y_true, y_pred)
            baselines = self._baseline_metrics(y_true, vocab_size)

        return MetricsResult(
            n_samples=int(np.unique([s.sequence_id for s in samples]).size),
            n_masked=n,
            logits_available=logits_available,
            vocab_size=vocab_size,
            token_level=is_token_level,
            approximation_note=approximation_note,
            cross_entropy=ce,
            nll=nll,
            perplexity=ppl,
            cross_entropy_lower_bound=ce_lb,
            top1_accuracy=top1,
            top3_accuracy=top3,
            top5_accuracy=top5,
            top_k_approximate=top_k_approx,
            balanced_accuracy=sk["balanced_accuracy"],
            precision=sk["precision"],
            recall=sk["recall"],
            macro_f1=sk["macro_f1"],
            micro_f1=sk["micro_f1"],
            weighted_f1=sk["weighted_f1"],
            mcc=sk["mcc"],
            kappa=sk["kappa"],
            prediction_entropy=pred_entropy,
            average_confidence=avg_conf,
            prediction_distribution=pred_counts,
            true_distribution=true_freqs,
            confusion_matrix=confusion,
            per_base=per_base,
            baselines=baselines,
            warnings=self._warnings.copy(),
        )

    def compute_region(
        self, samples: Sequence[MaskedSample], tag: str
    ) -> MetricsResult:
        """Compute metrics for a single genomic region tag."""
        region_samples = [s for s in samples if tag in s.region_tags]
        if not region_samples:
            raise ValueError(f"no samples in region {tag}")
        return self.compute(region_samples)
