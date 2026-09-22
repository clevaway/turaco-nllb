import pandas as pd

from turaco_nllb.data import (
    clean_parallel_dataframe,
    detokenize_mt560,
    leakage_resistant_split,
    normalize_for_match,
    strip_leading_reference_noise,
    verify_no_split_overlap,
)


def test_detokenize_repairs_common_mt560_spacing():
    assert detokenize_mt560("I don ' t know .") == "I don't know."
    assert detokenize_mt560("Wait , what ?") == "Wait, what?"
    assert detokenize_mt560("state - owned") == "state-owned"


def test_reference_cleaning_only_removes_leading_chapter_verse_notes():
    assert strip_leading_reference_noise("(John 3:16) God loves the world.") == "God loves the world."
    assert strip_leading_reference_noise("We read (John 3:16) today.") == "We read (John 3:16) today."


def test_cleaner_deduplicates_and_preserves_audit_rows():
    frame = pd.DataFrame(
        {
            "eng": ["How are you ?", "How are you?", "A", "This sentence is long enough"],
            "wes": ["How you dey ?", "How you dey?", "B", "Dis tok long reach well"],
        }
    )
    clean, rejected, report = clean_parallel_dataframe(frame)
    assert len(clean) == 2
    assert len(rejected) == 2
    assert report["rejection_counts"]["normalized_duplicate"] == 1
    assert normalize_for_match(clean.iloc[0].eng) == "how are you"


def test_split_keeps_duplicate_source_or_target_groups_together():
    frame = pd.DataFrame(
        {
            "eng": [
                "same source words",
                "same source words",
                "another source words",
                "last source words",
            ]
            * 40,
            "wes": [
                "target version one",
                "target version two",
                "shared target phrase",
                "shared target phrase",
            ]
            * 40,
        }
    )
    frame["eng"] = [f"{text} {index // 4}" for index, text in enumerate(frame.eng)]
    frame["wes"] = [f"{text} {index // 4}" for index, text in enumerate(frame.wes)]
    frame["eng_norm"] = frame.eng.map(normalize_for_match)
    frame["wes_norm"] = frame.wes.map(normalize_for_match)
    split = leakage_resistant_split(frame, seed=17)
    verify_no_split_overlap(split)
    assert set(split.split) == {"train", "validation", "test"}
    repeated = leakage_resistant_split(frame, seed=17)
    assert split.split.tolist() == repeated.split.tolist()

