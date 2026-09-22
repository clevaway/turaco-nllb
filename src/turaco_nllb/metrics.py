from __future__ import annotations

import math
import random
import statistics
from typing import Sequence

import sacrebleu

from .data import normalize_for_match


def corpus_mt_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    sources: Sequence[str] | None = None,
) -> dict[str, int | float | str]:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        raise ValueError("At least one prediction is required")

    bleu_metric = sacrebleu.metrics.BLEU(tokenize="13a", effective_order=True)
    chrf_metric = sacrebleu.metrics.CHRF(word_order=2)
    ter_metric = sacrebleu.metrics.TER()
    bleu = bleu_metric.corpus_score(list(predictions), [list(references)])
    chrf = chrf_metric.corpus_score(list(predictions), [list(references)])
    ter = ter_metric.corpus_score(list(predictions), [list(references)])

    exact = statistics.mean(
        normalize_for_match(prediction) == normalize_for_match(reference)
        for prediction, reference in zip(predictions, references)
    )
    prediction_lengths = [max(1, len(item.split())) for item in predictions]
    reference_lengths = [max(1, len(item.split())) for item in references]
    length_ratios = [p / r for p, r in zip(prediction_lengths, reference_lengths)]

    result: dict[str, int | float | str] = {
        "count": len(predictions),
        "chrf_plus_plus": round(float(chrf.score), 4),
        "sacrebleu": round(float(bleu.score), 4),
        "ter": round(float(ter.score), 4),
        "exact_match_percent": round(exact * 100, 4),
        "mean_output_reference_word_ratio": round(statistics.mean(length_ratios), 4),
        "chrf_signature": str(chrf_metric.get_signature()),
        "bleu_signature": str(bleu_metric.get_signature()),
        "ter_signature": str(ter_metric.get_signature()),
    }
    if sources is not None:
        if len(sources) != len(predictions):
            raise ValueError("sources and predictions must have the same length")
        copy_rate = statistics.mean(
            normalize_for_match(prediction) == normalize_for_match(source)
            for prediction, source in zip(predictions, sources)
        )
        result["source_copy_percent"] = round(copy_rate * 100, 4)
    return result


def bootstrap_confidence_interval(
    predictions: Sequence[str],
    references: Sequence[str],
    metric: str = "chrf_plus_plus",
    samples: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, int | float]:
    if samples < 20:
        raise ValueError("Use at least 20 bootstrap samples")
    if len(predictions) != len(references) or not predictions:
        raise ValueError("predictions and references must be non-empty and aligned")
    if metric not in {"chrf_plus_plus", "sacrebleu", "ter"}:
        raise ValueError(f"Unsupported bootstrap metric: {metric}")

    rng = random.Random(seed)
    count = len(predictions)
    scores: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(count) for _ in range(count)]
        sample_predictions = [predictions[index] for index in indices]
        sample_references = [references[index] for index in indices]
        score = corpus_mt_metrics(sample_predictions, sample_references)[metric]
        scores.append(float(score))

    scores.sort()
    alpha = (1.0 - confidence) / 2.0
    lower_index = max(0, math.floor(alpha * (samples - 1)))
    upper_index = min(samples - 1, math.ceil((1.0 - alpha) * (samples - 1)))
    return {
        "estimate": float(corpus_mt_metrics(predictions, references)[metric]),
        "lower": round(scores[lower_index], 4),
        "upper": round(scores[upper_index], 4),
        "confidence": confidence,
        "samples": samples,
    }


def paired_bootstrap_difference(
    predictions_a: Sequence[str],
    predictions_b: Sequence[str],
    references: Sequence[str],
    metric: str = "chrf_plus_plus",
    samples: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, int | float | str]:
    """Paired bootstrap interval for score(A) minus score(B) on aligned rows."""
    if not predictions_a or not (
        len(predictions_a) == len(predictions_b) == len(references)
    ):
        raise ValueError("Both prediction streams and references must be non-empty and aligned")
    if samples < 20:
        raise ValueError("Use at least 20 bootstrap samples")
    if metric not in {"chrf_plus_plus", "sacrebleu", "ter"}:
        raise ValueError(f"Unsupported bootstrap metric: {metric}")

    point_a = float(corpus_mt_metrics(predictions_a, references)[metric])
    point_b = float(corpus_mt_metrics(predictions_b, references)[metric])
    rng = random.Random(seed)
    count = len(references)
    differences: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(count) for _ in range(count)]
        refs = [references[index] for index in indices]
        sample_a = [predictions_a[index] for index in indices]
        sample_b = [predictions_b[index] for index in indices]
        score_a = float(corpus_mt_metrics(sample_a, refs)[metric])
        score_b = float(corpus_mt_metrics(sample_b, refs)[metric])
        differences.append(score_a - score_b)

    sorted_differences = sorted(differences)
    alpha = (1.0 - confidence) / 2.0
    lower_index = max(0, math.floor(alpha * (samples - 1)))
    upper_index = min(samples - 1, math.ceil((1.0 - alpha) * (samples - 1)))
    non_positive = sum(value <= 0 for value in differences) / samples
    non_negative = sum(value >= 0 for value in differences) / samples
    return {
        "metric": metric,
        "count": count,
        "score_a": round(point_a, 4),
        "score_b": round(point_b, 4),
        "difference_a_minus_b": round(point_a - point_b, 4),
        "lower": round(sorted_differences[lower_index], 4),
        "upper": round(sorted_differences[upper_index], 4),
        "confidence": confidence,
        "two_sided_p_approx": round(min(1.0, 2 * min(non_positive, non_negative)), 6),
        "samples": samples,
        "interpretation": "higher_is_better" if metric != "ter" else "lower_is_better",
    }
