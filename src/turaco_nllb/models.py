from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .constants import DIRECTIONS, ENG_LANG, SUPPORTED_BACKENDS, WES_LANG


def format_parallel_batch(
    source_texts: list[str],
    target_texts: list[str],
    backend: str,
    direction: str,
) -> tuple[list[str], list[str]]:
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}")
    if direction not in DIRECTIONS:
        raise ValueError(f"Unsupported direction: {direction}")
    spec = DIRECTIONS[direction]
    if backend == "marian":
        source_texts = [f"{spec['marian_target_tag']} {text}" for text in source_texts]
    elif backend == "byt5":
        source_texts = [f"{spec['byt5_prefix']}{text}" for text in source_texts]
    return source_texts, target_texts


def _train_nllb_sentencepiece_extension(
    base_tokenizer,
    texts: Sequence[str],
    vocabulary_size: int,
    minimum_character_frequency: int,
    output_directory: str,
) -> tuple[str, list[str]]:
    """Append corpus pieces to NLLB's SentencePiece model without deleting its vocabulary."""
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as sentencepiece_proto

    corpus_path = Path(output_directory) / "wes_tokenizer_corpus.txt"
    corpus_lines = [str(text).strip() for text in texts if str(text).strip()]
    if not corpus_lines:
        raise ValueError("Tokenizer extension needs at least one non-empty training sentence")
    corpus_path.write_text("\n".join(corpus_lines) + "\n", encoding="utf-8")

    character_counts = Counter("".join(corpus_lines))
    required_characters = "".join(
        character
        for character, count in character_counts.items()
        if not character.isspace() and count >= minimum_character_frequency
    )
    extension_prefix = str(Path(output_directory) / "wes_extension")
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=extension_prefix,
        vocab_size=int(vocabulary_size),
        model_type="unigram",
        character_coverage=1.0,
        required_chars=required_characters,
        hard_vocab_limit=False,
        add_dummy_prefix=False,
        max_sentencepiece_length=64,
        pad_id=0,
        eos_id=1,
        unk_id=2,
        bos_id=-1,
    )

    trained = sentencepiece_proto.ModelProto()
    trained.ParseFromString(Path(f"{extension_prefix}.model").read_bytes())
    merged = sentencepiece_proto.ModelProto()
    merged.ParseFromString(base_tokenizer.sp_model.serialized_model_proto())
    existing = {piece.piece for piece in merged.pieces}
    added_pieces: list[str] = []
    score_floor = min(piece.score for piece in merged.pieces)
    for piece in trained.pieces:
        if int(piece.type) != int(sentencepiece_proto.ModelProto.SentencePiece.NORMAL):
            continue
        if piece.piece in existing:
            continue
        appended = merged.pieces.add()
        appended.piece = piece.piece
        appended.score = score_floor + float(piece.score)
        existing.add(piece.piece)
        added_pieces.append(piece.piece)

    merged_path = Path(output_directory) / "merged_sentencepiece.bpe.model"
    merged_path.write_bytes(merged.SerializeToString())
    return str(merged_path), added_pieces


def _initialize_nllb_embeddings(
    model,
    base_tokenizer,
    tokenizer,
    added_pieces: Sequence[str],
) -> None:
    """Preserve shifted language rows and warm-start every newly introduced row."""
    import torch
    from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES

    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    old_input = input_layer.weight.detach().clone()
    output_is_tied = output_layer is None or output_layer.weight.data_ptr() == input_layer.weight.data_ptr()
    old_output = None if output_is_tied else output_layer.weight.detach().clone()

    if len(tokenizer) > input_layer.num_embeddings:
        try:
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        except TypeError:
            model.resize_token_embeddings(len(tokenizer))
        input_layer = model.get_input_embeddings()
        output_layer = model.get_output_embeddings()

    preserved_tokens = list(FAIRSEQ_LANGUAGE_CODES)
    if base_tokenizer.mask_token:
        preserved_tokens.append(base_tokenizer.mask_token)

    with torch.no_grad():
        for token in preserved_tokens:
            old_id = base_tokenizer.convert_tokens_to_ids(token)
            new_id = tokenizer.convert_tokens_to_ids(token)
            if old_id == base_tokenizer.unk_token_id or new_id == tokenizer.unk_token_id:
                continue
            input_layer.weight[new_id].copy_(old_input[old_id])
            if old_output is not None and output_layer is not None:
                output_layer.weight[new_id].copy_(old_output[old_id])

        for piece in added_pieces:
            new_id = tokenizer.convert_tokens_to_ids(piece)
            old_ids = base_tokenizer(piece, add_special_tokens=False)["input_ids"]
            usable_ids = [index for index in old_ids if 0 <= index < old_input.shape[0]]
            if not usable_ids:
                usable_ids = [base_tokenizer.unk_token_id]
            input_layer.weight[new_id].copy_(old_input[usable_ids].mean(dim=0))
            if old_output is not None and output_layer is not None:
                output_layer.weight[new_id].copy_(old_output[usable_ids].mean(dim=0))

        wes_id = tokenizer.convert_tokens_to_ids(WES_LANG)
        related_ids = [
            tokenizer.convert_tokens_to_ids(ENG_LANG),
            tokenizer.convert_tokens_to_ids("tpi_Latn"),
        ]
        input_layer.weight[wes_id].copy_(input_layer.weight[related_ids].mean(dim=0))
        if old_output is not None and output_layer is not None:
            output_layer.weight[wes_id].copy_(output_layer.weight[related_ids].mean(dim=0))
    model.tie_weights()


def create_tokenizer_and_model(
    base_model: str,
    backend: str,
    extension_texts: Sequence[str] | None = None,
    tokenizer_extension: Mapping[str, Any] | None = None,
):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}")

    setup_metadata: dict[str, Any] = {}
    if backend == "nllb":
        from transformers import NllbTokenizer
        from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES

        language_codes = list(FAIRSEQ_LANGUAGE_CODES)
        if WES_LANG not in language_codes:
            language_codes.append(WES_LANG)
        base_tokenizer = NllbTokenizer.from_pretrained(
            base_model, src_lang=ENG_LANG, tgt_lang=ENG_LANG, use_fast=False
        )
        extension_config = dict(tokenizer_extension or {})
        extension_enabled = bool(extension_config.get("enabled", False))
        added_pieces: list[str] = []
        if extension_enabled:
            if extension_texts is None:
                raise ValueError("extension_texts are required when tokenizer extension is enabled")
            with tempfile.TemporaryDirectory(prefix="turaco_nllb_tokenizer_") as temporary_dir:
                merged_model, added_pieces = _train_nllb_sentencepiece_extension(
                    base_tokenizer,
                    list(extension_texts),
                    vocabulary_size=int(extension_config.get("vocab_size", 4_000)),
                    minimum_character_frequency=int(extension_config.get("min_character_frequency", 3)),
                    output_directory=temporary_dir,
                )
                base_tokenizer.save_pretrained(temporary_dir)
                shutil.copyfile(
                    merged_model,
                    os.path.join(temporary_dir, "sentencepiece.bpe.model"),
                )
                tokenizer = NllbTokenizer.from_pretrained(
                    temporary_dir,
                    additional_special_tokens=language_codes,
                    src_lang=ENG_LANG,
                    tgt_lang=WES_LANG,
                    use_fast=False,
                )
        else:
            tokenizer = NllbTokenizer.from_pretrained(
                base_model,
                additional_special_tokens=language_codes,
                src_lang=ENG_LANG,
                tgt_lang=WES_LANG,
                use_fast=False,
            )
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        new_token_id = tokenizer.convert_tokens_to_ids(WES_LANG)
        if new_token_id == tokenizer.unk_token_id or tokenizer.convert_ids_to_tokens(new_token_id) != WES_LANG:
            raise RuntimeError("wes_Latn was not added to the NLLB tokenizer")
        _initialize_nllb_embeddings(model, base_tokenizer, tokenizer, added_pieces)
        model.config.decoder_start_token_id = tokenizer.eos_token_id
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.decoder_start_token_id = tokenizer.eos_token_id
        setup_metadata = {
            "language_token": WES_LANG,
            "language_token_id": int(new_token_id),
            "sentencepiece_extension_enabled": extension_enabled,
            "sentencepiece_pieces_added": len(added_pieces),
            "tokenizer_size": len(tokenizer),
        }
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        if backend == "marian":
            unknown_tags = [
                tag
                for tag in {spec["marian_target_tag"] for spec in DIRECTIONS.values()}
                if tokenizer.convert_tokens_to_ids(tag) == tokenizer.unk_token_id
            ]
            if unknown_tags:
                raise RuntimeError(f"Marian tokenizer is missing target tags: {unknown_tags}")

    return tokenizer, model, setup_metadata


def apply_lora(model, backend: str, rank: int = 16, alpha: int = 32, dropout: float = 0.05):
    from peft import LoraConfig, TaskType, get_peft_model

    target_modules = ["q", "v"] if backend == "byt5" else ["q_proj", "v_proj"]
    config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    return get_peft_model(model, config)


def tokenize_direction(
    dataset,
    tokenizer,
    backend: str,
    direction: str,
    max_source_length: int,
    max_target_length: int,
):
    spec = DIRECTIONS[direction]

    def preprocess(batch: dict[str, list[str]]) -> dict[str, Any]:
        sources = list(batch[spec["source_column"]])
        targets = list(batch[spec["target_column"]])
        sources, targets = format_parallel_batch(sources, targets, backend, direction)
        if backend == "nllb":
            tokenizer.src_lang = spec["source_lang"]
            tokenizer.tgt_lang = spec["target_lang"]
        encoded = tokenizer(
            sources,
            text_target=targets,
            max_length=max_source_length,
            truncation=True,
        )
        if max_target_length != max_source_length:
            target_tokens = tokenizer(
                text_target=targets,
                max_length=max_target_length,
                truncation=True,
            )
            encoded["labels"] = target_tokens["input_ids"]
        return encoded

    removable = list(dataset.column_names)
    return dataset.map(preprocess, batched=True, remove_columns=removable)


def generation_inputs(texts: list[str], tokenizer, backend: str, direction: str, **tokenizer_kwargs):
    spec = DIRECTIONS[direction]
    formatted, _ = format_parallel_batch(texts, [""] * len(texts), backend, direction)
    if backend == "nllb":
        tokenizer.src_lang = spec["source_lang"]
    return tokenizer(formatted, **tokenizer_kwargs)


def generate_translations(
    model,
    tokenizer,
    texts: Iterable[str],
    backend: str,
    direction: str,
    batch_size: int = 16,
    num_beams: int = 5,
    max_source_length: int = 192,
    max_new_tokens: int = 192,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 3,
    repetition_penalty: float = 1.1,
) -> list[str]:
    import torch

    all_texts = list(texts)
    predictions: list[str] = []
    model.eval()
    device = next(model.parameters()).device
    spec = DIRECTIONS[direction]

    for start in range(0, len(all_texts), batch_size):
        batch = all_texts[start : start + batch_size]
        encoded = generation_inputs(
            batch,
            tokenizer,
            backend,
            direction,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_source_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generation_kwargs: dict[str, Any] = {
            "num_beams": num_beams,
            "max_new_tokens": max_new_tokens,
            "length_penalty": length_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "repetition_penalty": repetition_penalty,
        }
        if backend == "nllb":
            generation_kwargs["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids(
                spec["target_lang"]
            )
            generation_kwargs["decoder_start_token_id"] = tokenizer.eos_token_id
            generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
            generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
        with torch.inference_mode():
            generated = model.generate(**encoded, **generation_kwargs)
        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return [prediction.strip() for prediction in predictions]


def save_runtime_metadata(
    output_dir: str | Path,
    backend: str,
    base_model: str,
    directions: list[str],
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "project": "Turaco-NLLB",
        "backend": backend,
        "base_model": base_model,
        "directions": directions,
        "source_language": ENG_LANG,
        "target_language": WES_LANG,
    }
    if extra:
        payload.update(extra)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "turaco_config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_runtime_metadata(model_path: str | Path) -> dict[str, Any]:
    config_path = Path(model_path) / "turaco_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} is missing. Supply a Turaco-NLLB checkpoint with runtime metadata."
        )
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_saved_model(model_path: str | Path):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, NllbTokenizer

    metadata = load_runtime_metadata(model_path)
    backend = metadata["backend"]
    tokenizer_class = NllbTokenizer if backend == "nllb" else AutoTokenizer
    tokenizer = tokenizer_class.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return tokenizer, model, metadata
