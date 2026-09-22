"""Validate approved translations and create immutable two-direction evaluation rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Completed translation workbook")
    parser.add_argument("--output", default="TuracoBench_v1.csv")
    args = parser.parse_args()

    frame = pd.read_csv(args.input).fillna("")
    required = {
        "id",
        "source",
        "reference",
        "translator_id",
        "reviewer_id",
        "adjudication_status",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.id.duplicated().any():
        raise ValueError("IDs must be unique")
    if (frame.source.astype(str).str.strip() == "").any():
        raise ValueError("Every row needs an English source")
    if (frame.reference.astype(str).str.strip() == "").any():
        raise ValueError("Every row needs an adjudicated Cameroon Pidgin reference")
    if not (frame.adjudication_status.astype(str).str.lower() == "approved").all():
        raise ValueError("Every row must have adjudication_status=approved")
    if (frame.translator_id.astype(str).str.strip() == "").any() or (
        frame.reviewer_id.astype(str).str.strip() == ""
    ).any():
        raise ValueError("Every row needs translator and reviewer provenance")

    metadata_columns = [
        column
        for column in frame.columns
        if column
        not in {
            "id",
            "direction",
            "source",
            "reference",
            "reference_2",
        }
    ]
    en_wes = pd.DataFrame(
        {
            "id": frame.id.astype(str) + "-en-wes",
            "direction": "en-wes",
            "source": frame.source,
            "reference": frame.reference,
        }
    )
    wes_en = pd.DataFrame(
        {
            "id": frame.id.astype(str) + "-wes-en",
            "direction": "wes-en",
            "source": frame.reference,
            "reference": frame.source,
        }
    )
    for column in metadata_columns:
        en_wes[column] = frame[column]
        wes_en[column] = frame[column]
    final = pd.concat([en_wes, wes_en], ignore_index=True)
    output_path = Path(args.output)
    final.to_csv(output_path, index=False)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    manifest = {
        "rows": len(final),
        "source_rows": len(frame),
        "directions": final.direction.value_counts().to_dict(),
        "sha256": digest,
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

