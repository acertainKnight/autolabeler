# Calibration Examples

These examples are shown to the jury models to calibrate their understanding of the label boundaries. Include 3-5 examples per label, focusing on the most common confusion points.

---

## [LABEL_A] Examples

**Example 1**
> "[Text that clearly belongs to LABEL_A]"

Label: **[LABEL_A]**
Reasoning: [One or two sentences explaining which rule applies and why this isn't borderline]

---

**Example 2**
> "[Text that belongs to LABEL_A, slightly less obvious]"

Label: **[LABEL_A]**
Reasoning: [Explain the key signal that makes this LABEL_A rather than LABEL_C]

---

## [LABEL_B] Examples

**Example 3**
> "[Text that clearly belongs to LABEL_B]"

Label: **[LABEL_B]**
Reasoning: [Explain]

---

## [LABEL_C] (Default) Examples

**Example 4**
> "[Text that looks like LABEL_A at first glance but is actually LABEL_C]"

Label: **[LABEL_C]**
Reasoning: [Explain why the apparent signal is absent, negated, or insufficient]

---

**Example 5**
> "[Another borderline LABEL_C case]"

Label: **[LABEL_C]**
Reasoning: [Explain]

---

## Borderline / Edge Cases

**Example 6** *(common mistake)*
> "[Text where annotators often disagree]"

Label: **[CORRECT_LABEL]**
Reasoning: [Explain the tiebreaker rule that resolves this. Reference the specific rule by name.]

---

**Example 7** *(common mistake)*
> "[Another frequently confused text]"

Label: **[CORRECT_LABEL]**
Reasoning: [Explain]
