from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


RELIGIOUS_MARKER_RE = re.compile(
    r"\b(?:jehovah|bible|jesus|christ|christian|god|apostle|congregation|"
    r"ministry|scripture|kingdom|worship|baptis\w*|disciple|preach\w*|"
    r"prayer|resurrection|satan)\b",
    flags=re.IGNORECASE,
)

LEADING_REFERENCE_RE = re.compile(
    r"^\s*(?:\((?=[^)]{0,100}\d+\s*:\s*\d+)[^)]{1,120}\)\s*)+"
)


@dataclass(frozen=True)
class CleaningConfig:
    min_words: int = 2
    max_words: int = 180
    min_word_ratio: float = 0.35
    max_word_ratio: float = 3.0
    strip_reference_noise: bool = True
    detokenize: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "CleaningConfig":
        if not values:
            return cls()
        known = {field: values[field] for field in asdict(cls()).keys() if field in values}
        return cls(**known)


def detokenize_mt560(text: str) -> str:
    """Undo conservative punctuation spacing found in the OPUS-derived corpus."""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"([({\[])\s+", r"\1", text)
    text = re.sub(r"\s+([)}\]])", r"\1", text)
    text = re.sub(r"\b([A-Za-z]+)\s+'\s*(s|t|re|ve|ll|d|m)\b", r"\1'\2", text, flags=re.I)
    text = re.sub(r"(?<=\w)\s+-\s+(?=\w)", "-", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_leading_reference_noise(text: str) -> str:
    """Remove one or more leading parenthetical chapter/verse citations."""
    previous = None
    while previous != text:
        previous = text
        text = LEADING_REFERENCE_RE.sub("", text).strip()
    return text


def normalize_for_match(text: str) -> str:
    text = detokenize_mt560(text)
    text = text.casefold()
    text = re.sub(r"[^\w\s'-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def stable_example_id(eng: str, wes: str) -> str:
    payload = f"{normalize_for_match(eng)}\0{normalize_for_match(wes)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    pairs = sorted(
        f"{normalize_for_match(row.eng)}\0{normalize_for_match(row.wes)}"
        for row in frame[["eng", "wes"]].itertuples(index=False)
    )
    return hashlib.sha256("\n".join(pairs).encode("utf-8")).hexdigest()


def audit_dataframe(frame: pd.DataFrame) -> dict[str, Any]:
    required = {"eng", "wes"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    eng = frame["eng"].fillna("").astype(str)
    wes = frame["wes"].fillna("").astype(str)
    eng_norm = eng.map(normalize_for_match)
    wes_norm = wes.map(normalize_for_match)
    eng_words = eng.str.split().str.len().clip(lower=1)
    wes_words = wes.str.split().str.len().clip(lower=1)
    ratios = wes_words / eng_words

    def numeric_summary(series: pd.Series) -> dict[str, float]:
        quantiles = series.quantile([0.01, 0.5, 0.95, 0.99])
        return {
            "mean": round(float(series.mean()), 3),
            "p01": round(float(quantiles.loc[0.01]), 3),
            "p50": round(float(quantiles.loc[0.5]), 3),
            "p95": round(float(quantiles.loc[0.95]), 3),
            "p99": round(float(quantiles.loc[0.99]), 3),
            "max": round(float(series.max()), 3),
        }

    source_target = pd.DataFrame({"eng": eng_norm, "wes": wes_norm})
    eng_conflicts = source_target.groupby("eng", dropna=False)["wes"].nunique()
    wes_conflicts = source_target.groupby("wes", dropna=False)["eng"].nunique()

    target_parenthetical_only = (~eng.str.match(r"^\s*\(")) & wes.str.match(r"^\s*\(")
    religious = eng.str.contains(RELIGIOUS_MARKER_RE)

    return {
        "rows": int(len(frame)),
        "null_eng": int(frame["eng"].isna().sum()),
        "null_wes": int(frame["wes"].isna().sum()),
        "blank_eng": int((eng.str.strip() == "").sum()),
        "blank_wes": int((wes.str.strip() == "").sum()),
        "exact_duplicate_pairs": int(frame.duplicated(["eng", "wes"]).sum()),
        "normalized_duplicate_pairs": int(pd.Series(list(zip(eng_norm, wes_norm))).duplicated().sum()),
        "duplicate_english_rows": int(eng_norm.duplicated().sum()),
        "duplicate_wes_rows": int(wes_norm.duplicated().sum()),
        "english_word_counts": numeric_summary(eng_words),
        "wes_word_counts": numeric_summary(wes_words),
        "wes_to_english_word_ratio": numeric_summary(ratios),
        "word_ratio_below_0_4": int((ratios < 0.4).sum()),
        "word_ratio_above_2_5": int((ratios > 2.5).sum()),
        "religious_marker_rows": int(religious.sum()),
        "religious_marker_percent": round(float(religious.mean() * 100), 2),
        "english_space_before_punctuation_rows": int(eng.str.contains(r"\s[,.;:!?]").sum()),
        "english_space_before_punctuation_percent": round(
            float(eng.str.contains(r"\s[,.;:!?]").mean() * 100), 2
        ),
        "target_only_leading_parenthetical_rows": int(target_parenthetical_only.sum()),
        "english_conflicting_target_groups": int((eng_conflicts > 1).sum()),
        "wes_conflicting_source_groups": int((wes_conflicts > 1).sum()),
        "fingerprint_sha256": dataframe_fingerprint(frame),
    }


def clean_parallel_dataframe(
    frame: pd.DataFrame,
    config: CleaningConfig | Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not isinstance(config, CleaningConfig):
        config = CleaningConfig.from_mapping(config)

    required = {"eng", "wes"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for source_index, row in frame[["eng", "wes"]].iterrows():
        eng_raw = "" if pd.isna(row.eng) else str(row.eng)
        wes_raw = "" if pd.isna(row.wes) else str(row.wes)
        eng = detokenize_mt560(eng_raw) if config.detokenize else eng_raw.strip()
        wes = detokenize_mt560(wes_raw) if config.detokenize else wes_raw.strip()

        if config.strip_reference_noise:
            eng = strip_leading_reference_noise(eng)
            wes = strip_leading_reference_noise(wes)

        eng_words = len(eng.split())
        wes_words = len(wes.split())
        ratio = wes_words / max(eng_words, 1)
        reasons: list[str] = []

        if not eng or not wes:
            reasons.append("empty")
        if eng_words < config.min_words or wes_words < config.min_words:
            reasons.append("too_short")
        if eng_words > config.max_words or wes_words > config.max_words:
            reasons.append("too_long")
        if ratio < config.min_word_ratio:
            reasons.append("ratio_too_low")
        if ratio > config.max_word_ratio:
            reasons.append("ratio_too_high")

        record = {
            "source_index": int(source_index) if isinstance(source_index, int) else str(source_index),
            "eng_raw": eng_raw,
            "wes_raw": wes_raw,
            "eng": eng,
            "wes": wes,
            "eng_norm": normalize_for_match(eng),
            "wes_norm": normalize_for_match(wes),
            "eng_words": eng_words,
            "wes_words": wes_words,
            "word_ratio": round(ratio, 6),
        }
        if reasons:
            record["rejection_reason"] = ",".join(sorted(set(reasons)))
            rejected.append(record)
        else:
            accepted.append(record)

    clean = pd.DataFrame(accepted)
    rejected_frame = pd.DataFrame(rejected)
    if clean.empty:
        raise ValueError("No examples remain after cleaning")

    duplicate_mask = clean.duplicated(["eng_norm", "wes_norm"], keep="first")
    if duplicate_mask.any():
        duplicate_rows = clean.loc[duplicate_mask].copy()
        duplicate_rows["rejection_reason"] = "normalized_duplicate"
        rejected_frame = pd.concat([rejected_frame, duplicate_rows], ignore_index=True)
        clean = clean.loc[~duplicate_mask].copy()

    clean["example_id"] = [stable_example_id(e, w) for e, w in zip(clean.eng, clean.wes)]
    clean["religious_domain_marker"] = clean.eng.str.contains(RELIGIOUS_MARKER_RE)
    clean = clean.reset_index(drop=True)
    rejected_frame = rejected_frame.reset_index(drop=True)

    report = {
        "config": asdict(config),
        "input_rows": int(len(frame)),
        "accepted_rows": int(len(clean)),
        "rejected_rows": int(len(rejected_frame)),
        "rejection_counts": (
            rejected_frame["rejection_reason"].value_counts().sort_index().to_dict()
            if not rejected_frame.empty
            else {}
        ),
        "clean_fingerprint_sha256": dataframe_fingerprint(clean),
    }
    return clean, rejected_frame, report


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def leakage_resistant_split(
    frame: pd.DataFrame,
    validation_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    if validation_ratio <= 0 or test_ratio <= 0 or validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio and test_ratio must be positive and sum to less than 1")

    output = frame.copy().reset_index(drop=True)
    if "eng_norm" not in output:
        output["eng_norm"] = output.eng.map(normalize_for_match)
    if "wes_norm" not in output:
        output["wes_norm"] = output.wes.map(normalize_for_match)
    if "example_id" not in output:
        output["example_id"] = [stable_example_id(e, w) for e, w in zip(output.eng, output.wes)]

    union_find = _UnionFind(len(output))
    first_seen: dict[str, int] = {}
    for index, row in output[["eng_norm", "wes_norm"]].iterrows():
        for key in (f"eng:{row.eng_norm}", f"wes:{row.wes_norm}"):
            if key in first_seen:
                union_find.union(index, first_seen[key])
            else:
                first_seen[key] = index

    roots = [union_find.find(index) for index in range(len(output))]
    members: dict[int, list[int]] = {}
    for index, root in enumerate(roots):
        members.setdefault(root, []).append(index)

    assignments: dict[int, str] = {}
    for root, indices in members.items():
        group_key = "|".join(sorted(output.loc[indices, "example_id"].astype(str)))
        digest = hashlib.sha256(f"{seed}:{group_key}".encode("utf-8")).hexdigest()
        score = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
        if score < test_ratio:
            split = "test"
        elif score < test_ratio + validation_ratio:
            split = "validation"
        else:
            split = "train"
        assignments[root] = split

    output["group_id"] = [hashlib.sha256(f"{seed}:{root}".encode()).hexdigest()[:12] for root in roots]
    output["split"] = [assignments[root] for root in roots]

    present = set(output.split)
    if present != {"train", "validation", "test"}:
        raise RuntimeError(f"Deterministic split produced missing partitions: {present}")

    verify_no_split_overlap(output)
    return output


def verify_no_split_overlap(frame: pd.DataFrame) -> None:
    for column in ("eng_norm", "wes_norm"):
        values = {
            split: set(part[column].dropna())
            for split, part in frame.groupby("split", sort=False)
        }
        names = sorted(values)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                overlap = values[left].intersection(values[right])
                if overlap:
                    raise AssertionError(
                        f"Leakage detected in {column} between {left} and {right}: {len(overlap)} values"
                    )


def save_data_artifacts(
    output_dir: str | Path,
    raw_audit: Mapping[str, Any],
    cleaning_report: Mapping[str, Any],
    split_frame: pd.DataFrame,
    rejected_frame: pd.DataFrame,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "data_audit.json").write_text(
        json.dumps(raw_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_path / "cleaning_report.json").write_text(
        json.dumps(cleaning_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    split_frame.to_parquet(output_path / "clean_splits.parquet", index=False)
    rejected_frame.to_parquet(output_path / "rejected_rows.parquet", index=False)
    manifest = {
        "rows_by_split": {key: int(value) for key, value in split_frame.split.value_counts().items()},
        "groups_by_split": {
            key: int(value)
            for key, value in split_frame.groupby("split")["group_id"].nunique().items()
        },
        "fingerprint_sha256": dataframe_fingerprint(split_frame),
    }
    (output_path / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

