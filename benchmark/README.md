# TuracoBench v1

TuracoBench is the external English ↔ Cameroon Pidgin evaluation set for the Turaco family. This repository contains the preparation tools and annotation protocol. The translated benchmark will be published separately after speaker review.

## Design

| Partition | Size | Use |
|---|---:|---|
| Pilot | 300 segments | Pipeline checks and one-time recipe/decoding selection |
| Final | 1,012 segments | Frozen public release comparison |

The recommended source is the English FLORES/FLORES+ `devtest` set. It covers news, educational, and travel material and is not part of MT560. Follow the source dataset's access and redistribution conditions.

The design follows the WMT26 Creole MT data requirements:

- translations or post-edits by competent native or proficient speakers;
- a dataset/data card;
- model evaluation demonstrating that the set behaves sensibly;
- chrF++ and statistical significance for model comparisons.

## Files

- `TuracoBench_v1_template.csv`: gold-creation worksheet
- `human_evaluation_template.csv`: blinded rating sheet
- `ANNOTATION_GUIDELINES.md`: translator, reviewer, and adjudicator instructions
- `prepare_flores_source.py`: creates a source-only worksheet
- `finalize_benchmark.py`: validates adjudicated rows and emits both directions

## Gold creation workflow

1. Obtain the English `devtest` source lawfully.
2. Run `prepare_flores_source.py` or map the source into the template.
3. Assign each row to a translator; store identities separately from public IDs.
4. Have a second qualified person review every translation.
5. Adjudicate all disputed, flagged, or materially changed rows.
6. Freeze the text, metadata, row order, and SHA-256 hash.
7. Search every training source for exact and normalized benchmark overlap.
8. Run `finalize_benchmark.py` to produce direction-specific evaluation rows.
9. Keep the final benchmark inaccessible to training, tokenizer learning, prompt development, and checkpoint selection.

## Evaluation contract

The evaluation CSV consumed by `turaco-evaluate` requires:

```text
id,direction,source,reference
```

Optional provenance columns are preserved in prediction outputs. Valid directions are `en-wes` and `wes-en`.

Headline results must identify:

- benchmark version and SHA-256 hash;
- model/checkpoint hash;
- decoding configuration;
- SacreBLEU signatures;
- number of examples;
- confidence interval method and resample count;
- human-review sample and reviewer qualifications.

## Contamination rule

If any benchmark source or reference appears in training data, remove it from the benchmark or retrain without the overlapping data. Report the overlap count even when it is zero.
