from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from datasets import Dataset, concatenate_datasets, load_dataset

from .constants import DIRECTIONS
from .data import (
    CleaningConfig,
    audit_dataframe,
    clean_parallel_dataframe,
    leakage_resistant_split,
    save_data_artifacts,
)
from .metrics import bootstrap_confidence_interval, corpus_mt_metrics
from .models import (
    apply_lora,
    create_tokenizer_and_model,
    generate_translations,
    save_runtime_metadata,
    tokenize_direction,
)


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {"run_name", "backend", "base_model", "dataset_id", "output_dir"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Config is missing required keys: {sorted(missing)}")
    directions = config.get("directions", ["en-wes", "wes-en"])
    unknown = set(directions).difference(DIRECTIONS)
    if unknown:
        raise ValueError(f"Unsupported directions: {sorted(unknown)}")
    return config


def _to_hf_dataset(frame: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(frame[["eng", "wes"]], preserve_index=False)


def _tokenize_partitions(split_frame, tokenizer, config):
    token_config = config.get("tokenization", {})
    max_source_length = int(token_config.get("max_source_length", 160))
    max_target_length = int(token_config.get("max_target_length", 192))
    backend = config["backend"]
    directions = list(config.get("directions", ["en-wes", "wes-en"]))
    seed = int(config.get("seed", 42))
    output = {}
    for split_name in ("train", "validation"):
        base = _to_hf_dataset(split_frame.loc[split_frame.split == split_name])
        directional = [
            tokenize_direction(
                base,
                tokenizer,
                backend,
                direction,
                max_source_length,
                max_target_length,
            )
            for direction in directions
        ]
        output[split_name] = concatenate_datasets(directional).shuffle(seed=seed)
    return output


def _evaluate_test_set(model, tokenizer, split_frame, config, output_dir: Path):
    test = split_frame.loc[split_frame.split == "test"].copy()
    generation = config.get("generation", {})
    token_config = config.get("tokenization", {})
    backend = config["backend"]
    results: dict[str, Any] = {}
    prediction_frames: list[pd.DataFrame] = []

    for direction in config.get("directions", ["en-wes", "wes-en"]):
        spec = DIRECTIONS[direction]
        sources = test[spec["source_column"]].tolist()
        references = test[spec["target_column"]].tolist()
        predictions = generate_translations(
            model,
            tokenizer,
            sources,
            backend,
            direction,
            batch_size=int(generation.get("batch_size", 16)),
            num_beams=int(generation.get("num_beams", 5)),
            max_source_length=int(token_config.get("max_source_length", 192)),
            max_new_tokens=int(
                generation.get("max_new_tokens", token_config.get("max_target_length", 192))
            ),
            length_penalty=float(generation.get("length_penalty", 1.0)),
            no_repeat_ngram_size=int(generation.get("no_repeat_ngram_size", 3)),
            repetition_penalty=float(generation.get("repetition_penalty", 1.1)),
        )
        metrics = corpus_mt_metrics(predictions, references, sources)
        metrics["chrf_95_ci"] = bootstrap_confidence_interval(
            predictions,
            references,
            metric="chrf_plus_plus",
            samples=200,
            seed=int(config.get("seed", 42)),
        )
        results[direction] = metrics
        prediction_frames.append(
            pd.DataFrame(
                {
                    "example_id": test.example_id.tolist(),
                    "direction": direction,
                    "source": sources,
                    "reference": references,
                    "prediction": predictions,
                }
            )
        )

    prediction_dir = output_dir / "evaluation"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        prediction_dir / "internal_test_predictions.csv", index=False
    )
    (prediction_dir / "internal_test_metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return results


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    import torch
    from transformers import (
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    seed = int(config.get("seed", 42))
    set_seed(seed)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    raw = load_dataset(config["dataset_id"], split="train").to_pandas()
    raw_audit = audit_dataframe(raw)
    cleaning_config = CleaningConfig.from_mapping(config.get("data"))
    clean, rejected, cleaning_report = clean_parallel_dataframe(raw, cleaning_config)
    data_config = config.get("data", {})
    split_frame = leakage_resistant_split(
        clean,
        validation_ratio=float(data_config.get("validation_ratio", 0.05)),
        test_ratio=float(data_config.get("test_ratio", 0.05)),
        seed=seed,
    )
    save_data_artifacts(output_dir / "data", raw_audit, cleaning_report, split_frame, rejected)

    training_rows = split_frame.loc[split_frame.split == "train"]
    extension_texts = training_rows.wes.astype(str).tolist() if config["backend"] == "nllb" else None
    tokenizer, model, model_setup = create_tokenizer_and_model(
        config["base_model"],
        config["backend"],
        extension_texts=extension_texts,
        tokenizer_extension=config.get("tokenizer_extension"),
    )
    training_config = config.get("training", {})
    mode = str(training_config.get("mode", "full")).lower()
    if mode == "lora":
        model = apply_lora(
            model,
            config["backend"],
            rank=int(training_config.get("lora_rank", 16)),
            alpha=int(training_config.get("lora_alpha", 32)),
            dropout=float(training_config.get("lora_dropout", 0.05)),
        )
    elif mode != "full":
        raise ValueError("training.mode must be 'full' or 'lora'")

    if bool(training_config.get("gradient_checkpointing", True)):
        model.config.use_cache = False

    tokenized = _tokenize_partitions(split_frame, tokenizer, config)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)

    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        run_name=config["run_name"],
        num_train_epochs=float(training_config.get("epochs", 4)),
        learning_rate=float(training_config.get("learning_rate", 3e-5)),
        per_device_train_batch_size=int(training_config.get("train_batch_size", 2)),
        per_device_eval_batch_size=int(training_config.get("eval_batch_size", 4)),
        gradient_accumulation_steps=int(training_config.get("gradient_accumulation_steps", 16)),
        gradient_checkpointing=bool(training_config.get("gradient_checkpointing", True)),
        fp16=bool(training_config.get("fp16", True)) and torch.cuda.is_available(),
        bf16=bool(training_config.get("bf16", False)) and torch.cuda.is_available(),
        optim=str(training_config.get("optimizer", "adafactor")),
        warmup_ratio=float(training_config.get("warmup_ratio", 0.05)),
        label_smoothing_factor=float(training_config.get("label_smoothing_factor", 0.1)),
        max_grad_norm=float(training_config.get("max_grad_norm", 1.0)),
        logging_strategy="steps",
        logging_steps=int(training_config.get("logging_steps", 50)),
        eval_strategy="steps",
        eval_steps=int(training_config.get("eval_steps", 250)),
        save_strategy="steps",
        save_steps=int(training_config.get("save_steps", 250)),
        save_total_limit=int(training_config.get("save_total_limit", 2)),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        save_safetensors=True,
        remove_unused_columns=True,
        dataloader_num_workers=int(training_config.get("dataloader_num_workers", 2)),
        report_to="none",
        seed=seed,
        data_seed=seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=int(training_config.get("early_stopping_patience", 4))
            )
        ],
    )
    train_result = trainer.train(resume_from_checkpoint=training_config.get("resume_from_checkpoint"))

    best_dir = output_dir / "best"
    if mode == "lora":
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(best_dir, safe_serialization=True)
        model = merged_model
    else:
        trainer.save_model(best_dir)
        model = trainer.model
    tokenizer.save_pretrained(best_dir)
    model.config.use_cache = True
    model.config.save_pretrained(best_dir)
    save_runtime_metadata(
        best_dir,
        backend=config["backend"],
        base_model=config["base_model"],
        directions=list(config.get("directions", ["en-wes", "wes-en"])),
        extra={
            "dataset_id": config["dataset_id"],
            "clean_data_fingerprint_sha256": cleaning_report["clean_fingerprint_sha256"],
            "training_mode": mode,
            "model_setup": model_setup,
        },
    )

    if torch.cuda.is_available():
        model.to("cuda")
    metrics = _evaluate_test_set(model, tokenizer, split_frame, config, output_dir)
    summary = {
        "train_metrics": train_result.metrics,
        "internal_test_metrics": metrics,
        "best_checkpoint": str(best_dir),
        "raw_rows": raw_audit["rows"],
        "clean_rows": cleaning_report["accepted_rows"],
        "model_setup": model_setup,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Turaco-NLLB translation checkpoint")
    parser.add_argument("--config", required=True, help="Path to a YAML configuration")
    arguments = parser.parse_args()
    summary = run_training(load_config(arguments.config))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
