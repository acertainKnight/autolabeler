# Program Generation Prompt (ALCHEmist-style)

This prompt is used by `scripts/generate_programs.py` to generate Python heuristic labeling functions from high-confidence labeled examples.

---

You are an expert at writing Python heuristic functions for text classification. Based on the labeled examples below, write Python functions that capture the patterns used to classify [TEXT_TYPE] into [LABEL] categories.

**Task:** [Brief description of what's being classified]

**Labels:**
- [LABEL_A]: [Description]
- [LABEL_B]: [Description]
- [LABEL_C]: [Description]

**High-confidence examples:**
{examples}

Write {n_programs} Python functions. Each function should:
1. Take a single string argument `text`
2. Return one of the label strings, or `None` if it cannot confidently classify the text
3. Use simple heuristics (keyword matching, regex, length checks) — no ML models
4. Be precise (high precision) even if it only covers a subset of cases (low coverage is fine)
5. Include a docstring explaining the pattern it captures

```python
def label_{name}(text: str) -> str | None:
    """
    [Describe the specific pattern this function captures, e.g.
    "Labels texts that explicitly mention rate hikes as LABEL_A."]
    """
    # your implementation here
    return None
```

Write each function as a standalone Python snippet. Focus on patterns that are reliable and generalizable, not patterns specific to individual examples.
