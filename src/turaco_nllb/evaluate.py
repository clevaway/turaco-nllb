from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .constants import DIRECTIONS
from .metrics import bootstrap_confidence_interval, corpus_mt_metrics
from .models import generate_translations, load_saved_model


def load_benchmark(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "reference" not in frame and "reference_1" in frame:
        frame = frame.rename(columns={"reference_1": "reference"})
    required = {"id", "direction", "source", "reference"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Benchmark is missing columns: {sorted(missing)}")
    unknown = set(frame.direction.dropna()).difference(DIRECTIONS)
    if unknown:
        raise ValueError(f"Benchmark has unsupported directions: {sorted(unknown)}")
    frame = frame.dropna(subset=["source", "reference", "direction"]).copy()
    if frame.empty:
        raise ValueError("Benchmark contains no completed rows")
    return frame


def evaluate_checkpoint(
    model_path: str | Path,
    benchmark_path: str | Path,
    output_dir: str | Path,
    batch_size: int = 16,
    num_beams: int = 5,
    max_source_length: int = 192,
    max_new_tokens: int = 192,
    bootstrap_samples: int = 1000,
) -> dict:
    import torch

    tokenizer, model, metadata = load_saved_model(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    frame = load_benchmark(benchmark_path)
    unsupported = set(frame.direction).difference(metadata.get("directions", []))
    if unsupported:
        raise ValueError(
            f"Checkpoint was not trained for benchmark directions: {sorted(unsupported)}"
        )
    output_parts = []
    results = {}

    for direction, part in frame.groupby("direction", sort=True):
        sources = part.source.astype(str).tolist()
        references = part.reference.astype(str).tolist()
        predictions = generate_translations(
            model,
            tokenizer,
            sources,
            metadata["backend"],
            direction,
            batch_size=batch_size,
            num_beams=num_beams,
            max_source_length=max_source_length,
            max_new_tokens=max_new_tokens,
        )
        metrics = corpus_mt_metrics(predictions, references, sources)
        metrics["chrf_95_ci"] = bootstrap_confidence_interval(
            predictions,
            references,
            metric="chrf_plus_plus",
            samples=bootstrap_samples,
        )
        results[direction] = metrics
        scored = part.copy()
        scored["prediction"] = predictions
        scored["model"] = str(model_path)
        output_parts.append(scored)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pd.concat(output_parts, ignore_index=True).to_csv(
        output_path / "benchmark_predictions.csv", index=False
    )
    payload = {
        "model": str(model_path),
        "benchmark": str(benchmark_path),
        "runtime_metadata": metadata,
        "metrics": results,
    }
    (output_path / "benchmark_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Turaco-NLLB on TuracoBench")
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--max-source-length", type=int, default=192)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    result = evaluate_checkpoint(
        args.model,
        args.benchmark,
        args.output_dir,
        args.batch_size,
        args.num_beams,
        args.max_source_length,
        args.max_new_tokens,
        args.bootstrap_samples,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
