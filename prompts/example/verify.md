# Cross-Verification Prompt

This prompt is used in Stage 5 (Cross-Verification) for uncertain or low-confidence labels. A different model family independently reviews the label.

---

You are verifying a label assigned by another model. Your job is to independently assess whether the label is correct, or whether it should be overridden.

**Text:** {text}

**Proposed label:** {proposed_label}
**Jury confidence:** {confidence}
**Jury reasoning:** {reasoning}

**Labels and their definitions:**
- [LABEL_A]: [Definition]
- [LABEL_B]: [Definition]
- [LABEL_C]: [Definition]

Review the proposed label independently. Do not anchor on it — form your own judgment first, then decide whether to agree or override.

Return a JSON object with:
- `agree`: true if the proposed label is correct, false to override
- `label`: the correct label (same as proposed if agree=true, your label if agree=false)
- `confidence`: your confidence in the final label (0.0-1.0)
- `reasoning`: brief explanation

Example response:
```json
{
  "agree": false,
  "label": "[LABEL_C]",
  "confidence": 0.8,
  "reasoning": "The proposed label [LABEL_A] is not supported because [reason]. The text actually [describes/implies] [LABEL_C] because [reason]."
}
```
