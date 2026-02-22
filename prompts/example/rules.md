# Classification Rules

## THE SCALE

- **[LABEL_A]**: [Full definition. What does it mean for a text to receive this label? What are the necessary and sufficient conditions?]

- **[LABEL_B]**: [Full definition. Include common edge cases and how they resolve.]

- **[LABEL_C] (DEFAULT)**: [Full definition. Make clear this is the catch-all / neutral / default when other labels don't clearly apply.]

---

## RULE 1: [RULE NAME — e.g. "The Intensity Test"]

[Describe the rule clearly with the key distinction it draws.]

**Examples:**
- "[Example text A]" → **[LABEL]** ([brief reason])
- "[Example text B]" → **[LABEL]** ([brief reason])
- "[Example text C]" → **[LABEL]** ([brief reason])

---

## RULE 2: [RULE NAME — e.g. "Explicit vs Implicit Signal"]

[Describe the rule.]

**Examples:**
- "[Example text]" → **[LABEL]** ([reason])
- "[Example text]" → **[LABEL]** ([reason])

---

## RULE 3: [RULE NAME — e.g. "Negation / Cancellation"]

[Describe how negation or cancellation of signals should be handled.]

**Examples:**
- "[Example with negation]" → **[LABEL]** ([reason])
- "[Example without negation]" → **[LABEL]** ([reason])

---

## RESOLVING AMBIGUITY

- When in doubt between [LABEL_A] and [LABEL_C]: **choose [LABEL_C]** (the default)
- When in doubt between [LABEL_A] and [LABEL_B]: [guidance]
- Mixed signals: [guidance for how to handle texts with signals pointing in multiple directions]

---

## COMMON PATTERNS TO WATCH FOR

### [Pattern 1 Name]
[Description of a frequently misclassified pattern and the correct approach]

- "[Example]" → **[CORRECT LABEL]** (not [WRONG LABEL], because [reason])
- "[Example]" → **[CORRECT LABEL]**

### [Pattern 2 Name]
[Description]

---

## LABEL QUICK REFERENCE

| Signal Type | Label | Notes |
|-------------|-------|-------|
| [Signal description] | [LABEL] | [Any caveats] |
| [Signal description] | [LABEL] | [Any caveats] |
| [Signal description] | [LABEL] | [Any caveats] |
| Ambiguous / unclear | [DEFAULT LABEL] | When in doubt |
