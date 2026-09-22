import pytest

from turaco_nllb.models import format_parallel_batch


def test_directional_formatting_for_baselines():
    source, target = format_parallel_batch(["Hello"], ["Ashia"], "marian", "en-wes")
    assert source == [">>wes<< Hello"]
    assert target == ["Ashia"]

    source, _ = format_parallel_batch(["Hello"], ["Ashia"], "byt5", "en-wes")
    assert source[0].startswith("translate English to Cameroon Pidgin:")


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        format_parallel_batch(["a"], ["b"], "unknown", "en-wes")

