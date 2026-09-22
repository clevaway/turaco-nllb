# MT560 Cameroon Pidgin Data Audit

Dataset: `michsethowusu/english-cameroon-pidgin_sentence-pairs_mt560`

Audit date: 22 September 2026

## Reproducibility

| Item | Value |
|---|---|
| Raw rows | 28,159 |
| Columns used | `eng`, `wes` |
| Raw normalized-pair fingerprint | `8fdd41c8e698a58bc11a7d54f440d83698fe777b5822ffee706c8fb92fbdab8f` |
| Default-clean fingerprint | `15ef7fbc533f463bea3e842fc791b7de5ec7d4ee1e597ab3d7e612a87ac6ebd4` |
| Cleaner implementation | `src/turaco_nllb/data.py` |

Fingerprints are SHA-256 hashes over sorted normalized English/WES pairs. They make data drift visible without redistributing the dataset.

## Raw corpus findings

| Check | Count | Share or note |
|---|---:|---|
| Null English / WES | 0 / 0 | None found |
| Blank English / WES | 0 / 0 | None found |
| Exact duplicate pairs after the first | 1 | Raw string match |
| Normalized duplicate pairs after the first | 31 | Case/punctuation-normalized match |
| Repeated normalized English rows | 275 | May include valid alternate translations |
| Repeated normalized WES rows | 127 | May include valid paraphrases or alignment artifacts |
| English groups with multiple WES values | 145 | Needs contextual review |
| WES groups with multiple English values | 79 | Needs contextual review |
| English rows with space before punctuation | 27,684 | 98.31% |
| Target-only leading parenthetical | 1,989 | Often citation or inserted context |
| Word ratio below 0.4 | 27 | Likely truncation or alignment risk |
| Word ratio above 2.5 | 371 | Likely expansion or alignment risk |
| Explicit religious-marker rows | 12,357 | 43.88% |

The religious-domain flag matches a documented list of words such as `Jehovah`, `Bible`, `Jesus`, `Christian`, `God`, `congregation`, `ministry`, and related forms in the English source. It is intentionally conservative. A row without a marker can still be religious, and a row with `god` can occur outside a religious document.

## Length distribution

| Statistic | English words | WES words | WES/English ratio |
|---|---:|---:|---:|
| Mean | 17.742 | 22.823 | 1.326 |
| 1st percentile | 4 | 5 | 0.654 |
| Median | 16 | 21 | 1.250 |
| 95th percentile | 34 | 44 | 2.000 |
| 99th percentile | 45 | 60 | 2.667 |
| Maximum | 86 | 116 | 5.000 |

## Default cleaning policy

The default cleaner:

1. normalizes Unicode with NFKC;
2. collapses excess whitespace;
3. repairs conservative punctuation, contraction, and hyphen spacing;
4. removes leading parenthetical chapter/verse citations;
5. rejects blank rows and rows shorter than two words;
6. rejects rows longer than 180 whitespace-delimited words;
7. keeps word-count ratios between 0.35 and 3.0;
8. removes normalized duplicate pairs while retaining an audit trail.

### Outcome

| Result | Rows |
|---|---:|
| Accepted | 27,847 |
| Rejected or deduplicated | 312 |

| Rejection label | Rows |
|---|---:|
| `normalized_duplicate` | 35 |
| `ratio_too_high` | 103 |
| `ratio_too_high,too_short` | 15 |
| `ratio_too_low` | 21 |
| `ratio_too_low,too_short` | 12 |
| `too_short` | 43 |
| `empty,ratio_too_low,too_short` after reference stripping | 83 |

Rejection is conservative filtering, not proof that a row is linguistically wrong. Every rejected row is saved with its reason for manual recovery or policy revision.

## Leakage-resistant internal split

The splitter connects rows when they share a normalized English source or WES target. Connected rows receive one deterministic partition, preventing exact source or target reuse across splits.

| Split | Rows | Connected groups |
|---|---:|---:|
| Train | 25,158 | 24,867 |
| Validation | 1,335 | 1,324 |
| Test | 1,354 | 1,342 |

This is still a same-corpus test and cannot support broad quality claims. It exists to catch training regressions and choose checkpoints before external evaluation.

## Remaining risks

- The source page does not provide document IDs, so near-duplicate or same-document leakage cannot be completely prevented.
- The strong religious skew can produce false fluency in that domain and weak performance elsewhere.
- Citation stripping is pattern-based and can remove useful content in rare cases.
- Word-ratio filtering is not a semantic alignment model.
- Cameroon Pidgin has meaningful spelling and dialect variation that automatic normalization can obscure.

