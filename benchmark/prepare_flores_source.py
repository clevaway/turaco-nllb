"""Create a source-only TuracoBench worksheet from the English FLORES devtest set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="facebook/flores")
    parser.add_argument("--config", default="eng_Latn")
    parser.add_argument("--split", default="devtest")
    parser.add_argument("--output", default="TuracoBench_v1_translation_workbook.csv")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    dataset = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        trust_remote_code=True,
    )
    frame = dataset.to_pandas()
    text_column = next(
        (name for name in ("sentence", "text", "eng_Latn") if name in frame.columns),
        None,
    )
    if text_column is None:
        raise ValueError(f"Could not identify the text column. Columns: {list(frame.columns)}")
    if args.limit:
        frame = frame.head(args.limit)

    output = pd.DataFrame(
        {
            "id": [f"turacobench-v1-{index + 1:04d}" for index in range(len(frame))],
            "direction": "en-wes",
            "source": frame[text_column].astype(str),
            "reference": "",
            "reference_2": "",
            "domain": frame["domain"].astype(str) if "domain" in frame else "",
            "source_origin": f"{args.dataset}:{args.config}:{args.split}",
            "translator_id": "",
            "reviewer_id": "",
            "adjudication_status": "",
            "notes": "",
        }
    )
    output_path = Path(args.output)
    output.to_csv(output_path, index=False)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    manifest = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "rows": len(output),
        "source_text_column": text_column,
        "workbook_sha256": digest,
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

