import pytest

from turaco_nllb.metrics import (
    bootstrap_confidence_interval,
    corpus_mt_metrics,
    paired_bootstrap_difference,
)


def test_identity_translation_has_perfect_reference_metrics():
    references = ["I am going home.", "What are you doing?"]
    metrics = corpus_mt_metrics(references, references, references)
    assert metrics["chrf_plus_plus"] == pytest.approx(100.0)
    assert metrics["sacrebleu"] == pytest.approx(100.0)
    assert metrics["ter"] == pytest.approx(0.0)
    assert metrics["exact_match_percent"] == pytest.approx(100.0)
    assert metrics["source_copy_percent"] == pytest.approx(100.0)


def test_bootstrap_is_deterministic():
    predictions = ["A B", "C D", "E F"]
    references = ["A B", "C X", "E F"]
    first = bootstrap_confidence_interval(predictions, references, samples=20, seed=9)
    second = bootstrap_confidence_interval(predictions, references, samples=20, seed=9)
    assert first == second


def test_metrics_reject_misaligned_inputs():
    with pytest.raises(ValueError):
        corpus_mt_metrics(["one"], ["one", "two"])


def test_paired_bootstrap_reports_aligned_difference():
    references = ["one two", "three four", "five six"]
    stronger = references
    weaker = ["one", "wrong", "five"]
    result = paired_bootstrap_difference(stronger, weaker, references, samples=20, seed=3)
    assert result["difference_a_minus_b"] > 0
    assert result["lower"] >= 0
