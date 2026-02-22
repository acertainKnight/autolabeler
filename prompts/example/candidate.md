# Candidate Annotation Prompt

This prompt is used in Stage 4 (Candidate Annotation) when the jury disagrees. The candidate model provides a soft probability distribution over labels rather than a single hard label.

---

You are resolving a labeling disagreement. The jury models disagree on how to classify the following text.

**Text:** {text}

**Jury votes:** {jury_votes}

Your task is to provide a probability distribution over all possible labels. Do not force a single answer — if the text is genuinely ambiguous, reflect that in the distribution.

**Labels:**
- [LABEL_A]: [Brief description]
- [LABEL_B]: [Brief description]
- [LABEL_C]: [Brief description]

Consider:
1. Which label(s) does the text most clearly match?
2. Is the text genuinely ambiguous, or is one label clearly correct?
3. What is the probability that each label is the correct one?

Return a JSON object with:
- `label`: your best single-label guess
- `confidence`: your confidence in that label (0.0-1.0)
- `soft_label`: probability distribution over all labels (must sum to 1.0)
- `reasoning`: brief explanation of your assessment

Example response:
```json
{
  "label": "[LABEL_C]",
  "confidence": 0.65,
  "soft_label": {"[LABEL_A]": 0.15, "[LABEL_B]": 0.20, "[LABEL_C]": 0.65},
  "reasoning": "The text contains [signal] which suggests [LABEL_C], but [secondary signal] creates ambiguity with [LABEL_B]."
}
```
