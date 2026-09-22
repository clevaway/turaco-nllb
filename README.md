# Turaco-NLLB-mt-en-wes

**Turaco-NLLB-mt-en-wes** is Clevaway's English-to-Cameroon-Pidgin translation model. It fine-tunes [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M) on English and Cameroon Pidgin sentence pairs from MT560.

The model uses `eng_Latn` for English and adds `wes_Latn` for Cameroon Pidgin. Its SentencePiece vocabulary is extended with subwords learned from the training split, and the original NLLB language embeddings are preserved when the vocabulary changes.

The reverse-direction model is named **Turaco-NLLB-mt-wes-en**.

## Intended Use

Turaco-NLLB is intended for:

- English to Cameroon Pidgin research and prototyping
- localization drafts reviewed by fluent speakers
- low-resource machine-translation experiments

Do not use the model for unattended medical, legal, emergency, immigration, financial, or other high-stakes translation. The NLLB license excludes commercial use.

## Quick Start

After the trained checkpoint is published:

```python
import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

MODEL_ID = "fotiecodes/Turaco-NLLB-mt-en-wes"

tokenizer = NllbTokenizer.from_pretrained(MODEL_ID, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

text = "What are you doing today?"
tokenizer.src_lang = "eng_Latn"
inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)

outputs = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.convert_tokens_to_ids("wes_Latn"),
    decoder_start_token_id=tokenizer.eos_token_id,
    max_new_tokens=128,
    num_beams=5,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

For local checkpoints, the supplied CLI reads the saved runtime metadata:

```bash
turaco-translate \
  --model outputs/Turaco-NLLB-mt-en-wes/best \
  --direction en-wes \
  --text "What are you doing today?"
```

## Training

The Google Colab notebook is the easiest entry point:

1. Open `notebooks/Turaco_NLLB_Colab.ipynb`.
2. Select a T4 GPU runtime.
3. Choose `en-wes` or `wes-en` before training.
4. Run the notebook from top to bottom.
5. Save the final checkpoint archive outside the temporary Colab runtime.

A T4 is suitable for smoke tests and constrained full fine-tuning, but a complete run can take several hours and free-runtime availability is not guaranteed.

Command-line training uses the same validated pipeline:

```bash
python -m pip install -e ".[test]"
turaco-train --config configs/nllb_600m_t4_en_wes.yaml
```

The default T4 recipe uses a micro-batch of 2, gradient accumulation, FP16, gradient checkpointing, Adafactor, label smoothing, early stopping, and a leakage-resistant internal split. The English-to-Pidgin and Pidgin-to-English checkpoints are trained separately because directional models are easier to inspect and evaluate.

## Training Data

The source dataset is [`michsethowusu/english-cameroon-pidgin_sentence-pairs_mt560`](https://huggingface.co/datasets/michsethowusu/english-cameroon-pidgin_sentence-pairs_mt560), an MT560/OPUS-derived table with 28,159 English (`eng`) and Cameroon Pidgin (`wes`) pairs under CC BY 4.0.

The default cleaner retains 27,847 of the 28,159 source pairs. The corpus is strongly skewed toward religious text, so it is not representative of everyday Cameroon Pidgin. Cleaning rules, rejection counts, split sizes, and data fingerprints are recorded in [`docs/DATA_AUDIT.md`](docs/DATA_AUDIT.md).

## Evaluation

The notebook reports chrF++, SacreBLEU, and TER on the grouped MT560 test split. These scores are useful for checking a training run but remain same-corpus measurements. External evaluation uses TuracoBench, a separately translated and speaker-reviewed test set described in [`benchmark/README.md`](benchmark/README.md).

Evaluate a completed benchmark with:

```bash
turaco-evaluate \
  --model outputs/Turaco-NLLB-mt-en-wes/best \
  --benchmark benchmark/TuracoBench_v1.csv \
  --output-dir outputs/Turaco-NLLB-mt-en-wes/turacobench
```

## Limitations

- Training data is narrow, strongly domain-skewed, and contains alignment noise.
- Cameroon Pidgin spelling varies by speaker, region, and context; a single reference may penalize valid alternatives.
- The internal split cannot establish state-of-the-art performance because every partition comes from the same source corpus.
- Vocabulary extension and the new language token make the released tokenizer inseparable from the checkpoint.
- The model may copy English, omit content, overproduce familiar religious phrasing, or normalize authentic variation.
- NLLB's CC BY-NC 4.0 license prevents this checkpoint from being presented as a commercially permissive release.

## Ethical Considerations

Cameroon Pidgin speakers should participate in benchmark creation, error review, release decisions, and documentation. Evaluators should preserve valid orthographic and dialect variation rather than treating one spelling as the only correct form. Public examples and test sets should be screened for personal data, offensive material, and licensing constraints.

## Repository

- `notebooks/`: self-contained Colab training and evaluation notebook
- `src/turaco_nllb/`: data, tokenizer, training, inference, and metrics code
- `configs/`: reproducible candidate configurations
- `benchmark/`: TuracoBench preparation and annotation tools
- `docs/`: data audit and training guide
- `tests/`: data, metric, and model tests

## Citation

```bibtex
@software{turaco_nllb_2026,
  author = {Clevaway},
  title = {Turaco-NLLB-mt-en-wes},
  year = {2026},
  url = {https://huggingface.co/fotiecodes/Turaco-NLLB-mt-en-wes}
}
```

## License

Turaco-NLLB is released under [CC BY-NC 4.0](https://spdx.org/licenses/CC-BY-NC-4.0). The MT560-derived training data is licensed separately under CC BY 4.0.
