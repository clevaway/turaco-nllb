# %% [markdown]
# # Turaco-NLLB: English ↔ Cameroon Pidgin
#
# This notebook trains one directional Turaco-NLLB checkpoint on a Google Colab T4.
#
# - `en-wes` produces **Turaco-NLLB-mt-en-wes**.
# - `wes-en` produces **Turaco-NLLB-mt-wes-en**.
#
# It adds a real `wes_Latn` token, optionally extends NLLB's SentencePiece vocabulary,
# preserves shifted language-token embeddings, cleans MT560, creates leakage-resistant
# internal splits, fine-tunes, evaluates, and saves the model plus tokenizer.
#
# The internal score is a development diagnostic. Do not call it a release benchmark.

# %%
!pip install -q "transformers>=4.57,<5" "datasets>=4.4,<6" "accelerate>=1.10,<2" "sacrebleu>=2.5,<3" "sentencepiece>=0.2,<0.3" "protobuf>=4.25,<7" "pandas>=2.1" "pyarrow>=15" "sacremoses>=0.1,<0.2"

# %% [markdown]
# ## 1. Configuration
#
# `QUICK_RUN=True` is a pipeline test, not a final model. A complete T4 run can take
# several hours. Colab availability and disconnect limits vary.

# %%
from pathlib import Path

DIRECTION = "en-wes"              # Change to "wes-en" for the reverse checkpoint.
QUICK_RUN = False                  # True: small smoke run. False: full training data.
EXTEND_SENTENCEPIECE = True        # Ablate with False after the first successful run.
EXTENSION_VOCAB_SIZE = 4_000
SEED = 42

BASE_MODEL = "facebook/nllb-200-distilled-600M"
DATASET_ID = "michsethowusu/english-cameroon-pidgin_sentence-pairs_mt560"
RUN_NAME = f"Turaco-NLLB-mt-{DIRECTION}"
OUTPUT_DIR = Path("/content") / RUN_NAME
HF_REPO_ID = f"fotiecodes/{RUN_NAME}"
HF_PRIVATE = True                   # Change to False only when the model card is release-ready.

MAX_SOURCE_LENGTH = 160
MAX_TARGET_LENGTH = 192
EPOCHS = 1 if QUICK_RUN else 4
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 16
LEARNING_RATE = 3e-5

assert DIRECTION in {"en-wes", "wes-en"}
print({"run": RUN_NAME, "direction": DIRECTION, "quick_run": QUICK_RUN})

# %% [markdown]
# ## 2. Imports and runtime check

# %%
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import sacrebleu
import torch
from datasets import Dataset, load_dataset
from sentencepiece import sentencepiece_model_pb2 as sp_pb2
import sentencepiece as spm
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    NllbTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES

set_seed(SEED)
if not torch.cuda.is_available():
    raise RuntimeError("Select Runtime > Change runtime type > T4 GPU before continuing.")
print("GPU:", torch.cuda.get_device_name(0))
print("Transformers:", __import__("transformers").__version__)
print("Torch:", torch.__version__)

# %% [markdown]
# ## 3. Load, audit, and clean MT560

# %%
RELIGIOUS_RE = re.compile(
    r"\b(?:jehovah|bible|jesus|christ|christian|god|apostle|congregation|"
    r"ministry|scripture|kingdom|worship|baptis\w*|disciple|preach\w*|"
    r"prayer|resurrection|satan)\b",
    re.I,
)
REFERENCE_RE = re.compile(r"^\s*(?:\((?=[^)]{0,100}\d+\s*:\s*\d+)[^)]{1,120}\)\s*)+")


def detokenize(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"([({\[])\s+", r"\1", text)
    text = re.sub(r"\s+([)}\]])", r"\1", text)
    text = re.sub(r"\b([A-Za-z]+)\s+'\s*(s|t|re|ve|ll|d|m)\b", r"\1'\2", text, flags=re.I)
    text = re.sub(r"(?<=\w)\s+-\s+(?=\w)", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_reference(text):
    previous = None
    while previous != text:
        previous = text
        text = REFERENCE_RE.sub("", text).strip()
    return text


def normalize_match(text):
    text = detokenize(text).casefold()
    text = re.sub(r"[^\w\s'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


source_raw = load_dataset(DATASET_ID, split="train").to_pandas()[["eng", "wes"]]
raw = source_raw.copy()
raw["eng"] = raw.eng.fillna("").map(detokenize).map(strip_reference)
raw["wes"] = raw.wes.fillna("").map(detokenize).map(strip_reference)
raw["eng_norm"] = raw.eng.map(normalize_match)
raw["wes_norm"] = raw.wes.map(normalize_match)
raw["eng_words"] = raw.eng.str.split().str.len()
raw["wes_words"] = raw.wes.str.split().str.len()
raw["word_ratio"] = raw.wes_words / raw.eng_words.clip(lower=1)

valid = (
    raw.eng.str.strip().ne("")
    & raw.wes.str.strip().ne("")
    & raw.eng_words.between(2, 180)
    & raw.wes_words.between(2, 180)
    & raw.word_ratio.between(0.35, 3.0)
)
clean = raw.loc[valid].drop_duplicates(["eng_norm", "wes_norm"]).reset_index(drop=True)
clean["example_id"] = [
    hashlib.sha256(f"{eng}\0{wes}".encode()).hexdigest()[:16]
    for eng, wes in zip(clean.eng_norm, clean.wes_norm)
]

audit = {
    "raw_rows": int(len(raw)),
    "clean_rows": int(len(clean)),
    "removed_rows": int(len(raw) - len(clean)),
    "religious_marker_rows": int(raw.eng.str.contains(RELIGIOUS_RE).sum()),
    "target_only_leading_parenthetical_before_cleaning": int(
        ((~source_raw.eng.str.match(r"^\s*\(")) & source_raw.wes.str.match(r"^\s*\(")).sum()
    ),
}
audit

# %% [markdown]
# ## 4. Leakage-resistant internal split
#
# Rows connected by the same normalized English or WES text stay together.

# %%
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


uf = UnionFind(len(clean))
seen = {}
for index, row in clean[["eng_norm", "wes_norm"]].iterrows():
    for key in ("en:" + row.eng_norm, "wes:" + row.wes_norm):
        if key in seen:
            uf.union(index, seen[key])
        else:
            seen[key] = index

roots = [uf.find(i) for i in range(len(clean))]
groups = {}
for index, root in enumerate(roots):
    groups.setdefault(root, []).append(index)

assignment = {}
for root, indices in groups.items():
    key = "|".join(sorted(clean.loc[indices, "example_id"]))
    score = int(hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest()[:16], 16) / 0xFFFFFFFFFFFFFFFF
    assignment[root] = "test" if score < 0.05 else "validation" if score < 0.10 else "train"

clean["split"] = [assignment[root] for root in roots]
if QUICK_RUN:
    train_small = clean[clean.split == "train"].sample(min(2_000, (clean.split == "train").sum()), random_state=SEED)
    validation_small = clean[clean.split == "validation"].head(300)
    test_small = clean[clean.split == "test"].head(300)
    clean = pd.concat([train_small, validation_small, test_small], ignore_index=True)

for column in ("eng_norm", "wes_norm"):
    sets = {name: set(part[column]) for name, part in clean.groupby("split")}
    assert not (sets["train"] & sets["validation"])
    assert not (sets["train"] & sets["test"])
print(clean.split.value_counts())

# %% [markdown]
# ## 5. Extend NLLB safely
#
# SentencePiece additions shift the numeric IDs of NLLB's language tokens. The code below
# restores every original language embedding at its new ID and initializes each new Pidgin
# piece from its old-tokenizer decomposition.

# %%
ENG_LANG = "eng_Latn"
WES_LANG = "wes_Latn"


def build_extended_tokenizer(base_model, wes_texts, extension_vocab_size=4_000):
    base_tokenizer = NllbTokenizer.from_pretrained(
        base_model, src_lang=ENG_LANG, tgt_lang=ENG_LANG, use_fast=False
    )
    language_codes = list(FAIRSEQ_LANGUAGE_CODES)
    if WES_LANG not in language_codes:
        language_codes.append(WES_LANG)

    if not EXTEND_SENTENCEPIECE:
        tokenizer = NllbTokenizer.from_pretrained(
            base_model,
            additional_special_tokens=language_codes,
            src_lang=ENG_LANG,
            tgt_lang=WES_LANG,
            use_fast=False,
        )
        return base_tokenizer, tokenizer, []

    temporary_dir = tempfile.mkdtemp(prefix="turaco_spm_")
    corpus_path = Path(temporary_dir) / "wes.txt"
    texts = [str(text).strip() for text in wes_texts if str(text).strip()]
    corpus_path.write_text("\n".join(texts) + "\n", encoding="utf-8")
    counts = Counter("".join(texts))
    required_chars = "".join(c for c, n in counts.items() if not c.isspace() and n >= 3)
    prefix = str(Path(temporary_dir) / "extension")
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=prefix,
        vocab_size=extension_vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        required_chars=required_chars,
        hard_vocab_limit=False,
        add_dummy_prefix=False,
        max_sentencepiece_length=64,
        pad_id=0,
        eos_id=1,
        unk_id=2,
        bos_id=-1,
    )

    trained = sp_pb2.ModelProto()
    trained.ParseFromString(Path(prefix + ".model").read_bytes())
    merged = sp_pb2.ModelProto()
    merged.ParseFromString(base_tokenizer.sp_model.serialized_model_proto())
    existing = {piece.piece for piece in merged.pieces}
    score_floor = min(piece.score for piece in merged.pieces)
    added = []
    for piece in trained.pieces:
        if int(piece.type) != 1 or piece.piece in existing:
            continue
        new_piece = merged.pieces.add()
        new_piece.piece = piece.piece
        new_piece.score = score_floor + float(piece.score)
        existing.add(piece.piece)
        added.append(piece.piece)

    merged_path = Path(temporary_dir) / "merged.model"
    merged_path.write_bytes(merged.SerializeToString())
    base_tokenizer.save_pretrained(temporary_dir)
    shutil.copyfile(merged_path, Path(temporary_dir) / "sentencepiece.bpe.model")
    tokenizer = NllbTokenizer.from_pretrained(
        temporary_dir,
        additional_special_tokens=language_codes,
        src_lang=ENG_LANG,
        tgt_lang=WES_LANG,
        use_fast=False,
    )
    shutil.rmtree(temporary_dir, ignore_errors=True)
    return base_tokenizer, tokenizer, added


def remap_and_initialize(model, base_tokenizer, tokenizer, added_pieces):
    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    old_input = input_layer.weight.detach().clone()
    tied = output_layer is None or output_layer.weight.data_ptr() == input_layer.weight.data_ptr()
    old_output = None if tied else output_layer.weight.detach().clone()
    if len(tokenizer) > input_layer.num_embeddings:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()

    preserved = list(FAIRSEQ_LANGUAGE_CODES) + ([base_tokenizer.mask_token] if base_tokenizer.mask_token else [])
    with torch.no_grad():
        for token in preserved:
            old_id = base_tokenizer.convert_tokens_to_ids(token)
            new_id = tokenizer.convert_tokens_to_ids(token)
            if old_id == base_tokenizer.unk_token_id or new_id == tokenizer.unk_token_id:
                continue
            input_layer.weight[new_id].copy_(old_input[old_id])
            if old_output is not None:
                output_layer.weight[new_id].copy_(old_output[old_id])

        for piece in added_pieces:
            new_id = tokenizer.convert_tokens_to_ids(piece)
            old_ids = base_tokenizer(piece, add_special_tokens=False)["input_ids"]
            old_ids = [index for index in old_ids if 0 <= index < old_input.shape[0]] or [base_tokenizer.unk_token_id]
            input_layer.weight[new_id].copy_(old_input[old_ids].mean(0))
            if old_output is not None:
                output_layer.weight[new_id].copy_(old_output[old_ids].mean(0))

        wes_id = tokenizer.convert_tokens_to_ids(WES_LANG)
        seeds = [tokenizer.convert_tokens_to_ids(ENG_LANG), tokenizer.convert_tokens_to_ids("tpi_Latn")]
        input_layer.weight[wes_id].copy_(input_layer.weight[seeds].mean(0))
        if old_output is not None:
            output_layer.weight[wes_id].copy_(output_layer.weight[seeds].mean(0))
    model.tie_weights()
    model.config.decoder_start_token_id = tokenizer.eos_token_id
    model.generation_config.decoder_start_token_id = tokenizer.eos_token_id


train_wes = clean.loc[clean.split == "train", "wes"].tolist()
base_tokenizer, tokenizer, added_pieces = build_extended_tokenizer(
    BASE_MODEL, train_wes, EXTENSION_VOCAB_SIZE
)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
remap_and_initialize(model, base_tokenizer, tokenizer, added_pieces)
wes_id = tokenizer.convert_tokens_to_ids(WES_LANG)
assert wes_id != tokenizer.unk_token_id and tokenizer.convert_ids_to_tokens(wes_id) == WES_LANG
print({"tokenizer_size": len(tokenizer), "pieces_added": len(added_pieces), "wes_token_id": wes_id})
del base_tokenizer

# %% [markdown]
# ## 6. Tokenize the selected direction

# %%
DIRECTION_SPEC = {
    "en-wes": {"source": "eng", "target": "wes", "src_lang": ENG_LANG, "tgt_lang": WES_LANG},
    "wes-en": {"source": "wes", "target": "eng", "src_lang": WES_LANG, "tgt_lang": ENG_LANG},
}[DIRECTION]


def to_dataset(frame):
    return Dataset.from_pandas(frame[["eng", "wes"]], preserve_index=False)


def preprocess(batch):
    tokenizer.src_lang = DIRECTION_SPEC["src_lang"]
    tokenizer.tgt_lang = DIRECTION_SPEC["tgt_lang"]
    model_inputs = tokenizer(
        list(batch[DIRECTION_SPEC["source"]]),
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
    )
    labels = tokenizer(
        text_target=list(batch[DIRECTION_SPEC["target"]]),
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized = {}
for split in ("train", "validation"):
    data = to_dataset(clean[clean.split == split])
    tokenized[split] = data.map(preprocess, batched=True, remove_columns=data.column_names)
print(tokenized)

# %% [markdown]
# ## 7. Fine-tune

# %%
model.config.use_cache = False
collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)
training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
    run_name=RUN_NAME,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    gradient_checkpointing=True,
    fp16=True,
    optim="adafactor",
    warmup_ratio=0.05,
    label_smoothing_factor=0.1,
    max_grad_norm=1.0,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=250 if not QUICK_RUN else 50,
    save_strategy="steps",
    save_steps=250 if not QUICK_RUN else 50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    predict_with_generate=False,
    report_to="none",
    seed=SEED,
    data_seed=SEED,
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=collator,
    processing_class=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
)
train_result = trainer.train()
train_result.metrics

# %% [markdown]
# ## 8. Save and run the internal diagnostic

# %%
BEST_DIR = OUTPUT_DIR / "best"
model = trainer.model
model.config.use_cache = True
trainer.save_model(BEST_DIR)
tokenizer.save_pretrained(BEST_DIR)
model.config.save_pretrained(BEST_DIR)

runtime_metadata = {
    "project": "Turaco-NLLB",
    "run_name": RUN_NAME,
    "backend": "nllb",
    "base_model": BASE_MODEL,
    "directions": [DIRECTION],
    "source_language": ENG_LANG,
    "target_language": WES_LANG,
    "sentencepiece_extension_enabled": EXTEND_SENTENCEPIECE,
    "sentencepiece_pieces_added": len(added_pieces),
    "quick_run": QUICK_RUN,
}
(BEST_DIR / "turaco_config.json").write_text(json.dumps(runtime_metadata, indent=2), encoding="utf-8")
(OUTPUT_DIR / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


@torch.inference_mode()
def translate_batch(texts, batch_size=16):
    model.eval().to("cuda")
    outputs = []
    tokenizer.src_lang = DIRECTION_SPEC["src_lang"]
    forced_id = tokenizer.convert_tokens_to_ids(DIRECTION_SPEC["tgt_lang"])
    for start in range(0, len(texts), batch_size):
        batch = [str(text) for text in texts[start : start + batch_size]]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SOURCE_LENGTH,
        ).to("cuda")
        generated = model.generate(
            **encoded,
            forced_bos_token_id=forced_id,
            decoder_start_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=MAX_TARGET_LENGTH,
            num_beams=5,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
        )
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return [text.strip() for text in outputs]


test = clean[clean.split == "test"].copy()
sources = test[DIRECTION_SPEC["source"]].tolist()
references = test[DIRECTION_SPEC["target"]].tolist()
predictions = translate_batch(sources)
chrf = sacrebleu.metrics.CHRF(word_order=2)
bleu = sacrebleu.metrics.BLEU(tokenize="13a", effective_order=True)
ter = sacrebleu.metrics.TER()
metrics = {
    "warning": "Internal same-corpus diagnostic; not a release benchmark",
    "count": len(predictions),
    "chrf_plus_plus": chrf.corpus_score(predictions, [references]).score,
    "sacrebleu": bleu.corpus_score(predictions, [references]).score,
    "ter": ter.corpus_score(predictions, [references]).score,
    "chrf_signature": str(chrf.get_signature()),
    "bleu_signature": str(bleu.get_signature()),
    "ter_signature": str(ter.get_signature()),
}
pd.DataFrame({"source": sources, "reference": references, "prediction": predictions}).to_csv(
    OUTPUT_DIR / "internal_test_predictions.csv", index=False
)
(OUTPUT_DIR / "internal_test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
metrics

# %% [markdown]
# ## 9. Try a sentence

# %%
sample = "As Christians, we must reject harmful teaching and search for the truth."
print("Input:", sample)
print("Translation:", translate_batch([sample], batch_size=1)[0])

# %% [markdown]
# ## 10. Preserve the checkpoint
#
# Mount Drive for a persistent copy. A 600M checkpoint is large, so browser download is
# usually less reliable than Drive or Hugging Face Hub.

# %%
from google.colab import drive
drive.mount("/content/drive")
DRIVE_DESTINATION = Path("/content/drive/MyDrive/Turaco") / RUN_NAME
if DRIVE_DESTINATION.exists():
    raise FileExistsError(f"Refusing to overwrite {DRIVE_DESTINATION}")
shutil.copytree(BEST_DIR, DRIVE_DESTINATION)
print("Saved to", DRIVE_DESTINATION)

# %% [markdown]
# ## 11. Push the model to Hugging Face
#
# This is the final step. It prompts for your Hugging Face token, creates the repository,
# writes a model card, and uploads the weights plus the exact extended tokenizer. The
# default is a private repository because the score above is an internal diagnostic.

# %%
from huggingface_hub import HfApi, create_repo, notebook_login

notebook_login()
create_repo(HF_REPO_ID, repo_type="model", private=HF_PRIVATE, exist_ok=True)

model_card = f"""---
license: cc-by-nc-4.0
base_model: {BASE_MODEL}
datasets:
- {DATASET_ID}
language:
- en
- wes
library_name: transformers
pipeline_tag: translation
tags:
- turaco
- nllb
- cameroon-pidgin
---

# {RUN_NAME}

{RUN_NAME} is part of the **Turaco** model family. It is a direction-specific NLLB
checkpoint for {DIRECTION} machine translation.

## Model Details

- Developed by: fotiecodes
- Architecture: NLLB encoder-decoder
- Base model: `{BASE_MODEL}`
- Fine-tuning: full supervised fine-tuning
- Direction: `{DIRECTION}`
- Language token: `wes_Latn` (added by this project)
- SentencePiece pieces added: {len(added_pieces)}
- License: CC BY-NC 4.0

## Training Data

The checkpoint was trained on cleaned pairs from `{DATASET_ID}`. Raw rows: {audit['raw_rows']}.
Rows used after default cleaning: {audit['clean_rows']}.

## Evaluation

The following is a same-corpus internal diagnostic, not a release benchmark:

| N | chrF++ | SacreBLEU | TER |
|---:|---:|---:|---:|
| {metrics['count']} | {metrics['chrf_plus_plus']:.2f} | {metrics['sacrebleu']:.2f} | {metrics['ter']:.2f} |

External TuracoBench results are pending. Do not describe this checkpoint as state of the art
until speaker-reviewed external evaluation is complete.

## Limitations

Training data is domain-skewed and noisy. The model can copy English, omit content, add
content, or mishandle valid Cameroon Pidgin spelling and dialect variation. It is not for
unattended high-stakes translation or commercial use.
"""
(BEST_DIR / "README.md").write_text(model_card, encoding="utf-8")

model.push_to_hub(HF_REPO_ID, safe_serialization=True, commit_message="Upload Turaco-NLLB model")
tokenizer.push_to_hub(HF_REPO_ID, commit_message="Upload extended Turaco-NLLB tokenizer")
HfApi().upload_file(
    path_or_fileobj=str(BEST_DIR / "README.md"),
    path_in_repo="README.md",
    repo_id=HF_REPO_ID,
    repo_type="model",
    commit_message="Add Turaco-NLLB model card",
)
print(f"Uploaded: https://huggingface.co/{HF_REPO_ID}")
