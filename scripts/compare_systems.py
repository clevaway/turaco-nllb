from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from turaco_nllb.metrics import paired_bootstrap_difference


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two aligned TuracoBench prediction files")
    parser.add_argument("--system-a", required=True)
    parser.add_argument("--system-b", required=True)
    parser.add_argument("--metric", choices=["chrf_plus_plus", "sacrebleu", "ter"], default="chrf_plus_plus")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    left = pd.read_csv(args.system_a)
    right = pd.read_csv(args.system_b)
    required = {"id", "direction", "reference", "prediction"}
    for name, frame in (("system A", left), ("system B", right)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    merged = left[list(required)].merge(
        right[list(required)],
        on=["id", "direction"],
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("Prediction files do not contain exactly the same IDs and directions")
    if not (merged.reference_a.astype(str) == merged.reference_b.astype(str)).all():
        raise ValueError("References differ between the prediction files")

    results = {}
    for direction, part in merged.groupby("direction", sort=True):
        results[direction] = paired_bootstrap_difference(
            part.prediction_a.astype(str).tolist(),
            part.prediction_b.astype(str).tolist(),
            part.reference_a.astype(str).tolist(),
            metric=args.metric,
            samples=args.samples,
            seed=args.seed,
        )
    payload = {
        "system_a": args.system_a,
        "system_b": args.system_b,
        "metric": args.metric,
        "results": results,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

