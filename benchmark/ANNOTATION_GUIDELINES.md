# TuracoBench Annotation Guidelines

## Roles

- **Translator:** produces a natural Cameroon Pidgin rendering from the English source without seeing model output.
- **Reviewer:** independently checks meaning, naturalness, spelling variation, and completeness.
- **Adjudicator:** resolves disagreements and approves the canonical reference.

One person may not serve as translator and sole reviewer for the same row. Record whether each contributor is a native or proficient speaker, their regional familiarity, and the variety name they use for the language.

## Translation brief

Translate the complete meaning, not the English word order.

- Preserve names, numbers, dates, negation, modality, and factual relations.
- Use Cameroon Pidgin that sounds natural to the intended audience.
- Do not copy English merely because it is understandable.
- Do not introduce explanations, moral framing, honorifics, or contextual facts absent from the source.
- Preserve uncertainty and ambiguity when the source is uncertain or ambiguous.
- Keep punctuation readable, but do not force Standard English spelling rules onto Pidgin.
- Flag content that lacks a stable or familiar Cameroon Pidgin rendering.
- Record valid alternative spellings in `reference_2` only when they are full-sentence alternatives, not word lists.

Machine translation may not create the first reference. If machine output is used for post-editing, disclose the system and mark the row as post-edited rather than human-original.

## Review checklist

For every row, answer:

1. Is every meaning-bearing source element represented?
2. Was any meaning added?
3. Are polarity, tense/aspect, person, number, named entities, and quantities correct?
4. Does the text read as Cameroon Pidgin rather than Nigerian Pidgin, Standard English, or an artificial mixture?
5. Is the chosen spelling internally understandable and consistent?
6. Could another competent speaker reasonably interpret the translation differently?
7. Does the row contain sensitive, personally identifying, or unsafe content requiring special handling?

Use `adjudication_status` values: `translated`, `reviewed`, `needs_adjudication`, or `approved`.

## Human model-output ratings

Ratings are performed blind. Randomize system IDs and output order for each reviewer.

### Adequacy

| Score | Anchor |
|---:|---|
| 5 | All meaning is preserved; no material addition or omission. |
| 4 | Meaning is correct with a small, non-critical imperfection. |
| 3 | Main idea survives, but at least one meaningful detail is wrong or missing. |
| 2 | Some related meaning remains, but major content is wrong or absent. |
| 1 | Unrelated, contradictory, or effectively untranslated. |

### Fluency

| Score | Anchor |
|---:|---|
| 5 | Fully natural and easy to understand. |
| 4 | Natural overall with a minor awkward phrase. |
| 3 | Understandable but repeatedly awkward. |
| 2 | Difficult to read; substantial repair is needed. |
| 1 | Not intelligible as a sentence. |

### Dialect fidelity

| Score | Anchor |
|---:|---|
| 5 | Clearly natural Cameroon Pidgin for the stated audience. |
| 4 | Cameroon Pidgin with minor cross-variety or English influence. |
| 3 | Mixed or generic West African Pidgin, but still locally understandable. |
| 2 | Mostly another variety or Standard English. |
| 1 | Not Cameroon Pidgin. |

Mark the following separately as `0` or `1`: omission, addition/hallucination, wrong negation, wrong named entity/number, source copy, offensive distortion, and unsafe high-stakes error.

## Quality control

- Double-rate at least 20% of the benchmark and all safety-critical items.
- Report Krippendorff's alpha or an appropriate ordinal agreement statistic.
- Discuss disagreements about valid regional and spelling variation before changing guidelines.
- Keep an immutable change log after the first freeze.
- Compensate contributors fairly and record consent for publication.

