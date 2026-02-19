# Candidate Annotation Prompt (for Jury Disagreements)

The jury disagreed on this headline. Instead of forcing a single label, list ALL PLAUSIBLE labels with your estimated probability for each.

## Task

Analyze why this headline is ambiguous and which rules might support different interpretations. Then output ALL labels that have reasonable support, with probability weights that sum to 1.0.

## Response Format

Return ONLY a JSON object with this structure:

```json
{
  "reasoning": "Brief analysis of why this is ambiguous and which rules apply to each interpretation",
  "candidates": [
    {
      "label": <integer>,
      "probability": <float between 0 and 1>,
      "rationale": "One sentence explaining support for this label"
    },
    {
      "label": <integer>,
      "probability": <float between 0 and 1>,
      "rationale": "One sentence explaining support for this label"
    }
  ],
  "primary_label": <integer (highest probability)>,
  "ambiguity_type": "adjacent_class | cross_boundary | rule_conflict"
}
```

## Requirements

- Probabilities must sum to 1.0
- Only include labels with probability ≥ 0.15
- Most ambiguous cases will have 2 candidates (e.g., {0: 0.6, 1: 0.4})
- For truly split cases, probabilities can be close (e.g., {0: 0.5, 1: 0.5})

## Ambiguity Types

- **adjacent_class**: Borderline between adjacent labels (e.g., 0 vs 1, -1 vs 0)
- **cross_boundary**: Ambiguity spans neutral boundary (e.g., -1 vs 1, skipping 0)
- **rule_conflict**: Different rules support different labels

## Common Sources of Ambiguity

1. **Intensity borderline**: Is "strong" plain description (0) or intensity-modified (1)?
2. **Implicit vs explicit**: Is concern about conditions (-1) or call for action (-2)?
3. **Negation ambiguity**: Does negation neutralize or create signal?
4. **Context-dependent phrases**: "above neutral", "patient approach" without clear context
5. **Mixed signals**: Headline contains both hawkish and dovish elements

## Examples

### Example 1: Adjacent Class (0 vs 1)
Headline: "LABOR MARKET REMAINS STRONG, WAGE PRESSURES BUILDING"

```json
{
  "reasoning": "Mixed signal: 'strong' alone would be neutral (0) per Rule 1, but 'wage pressures building' suggests intensity and inflation risk (1). The combination creates ambiguity.",
  "candidates": [
    {
      "label": 0,
      "probability": 0.45,
      "rationale": "'Strong' without intensity modifier should be neutral per Rule 1"
    },
    {
      "label": 1,
      "probability": 0.55,
      "rationale": "'Wage pressures building' suggests inflation risk requiring restrictive policy"
    }
  ],
  "primary_label": 1,
  "ambiguity_type": "adjacent_class"
}
```

### Example 2: Rule Conflict (-1 vs 0)
Headline: "RATES ARE RESTRICTIVE; POLICY IS WORKING"

```json
{
  "reasoning": "'Rates are restrictive' could signal dovish lean (rates high), but 'policy is working' indicates satisfaction with current stance (neutral). Critical guidance says this combination should be neutral.",
  "candidates": [
    {
      "label": -1,
      "probability": 0.35,
      "rationale": "'Rates are restrictive' alone could imply dovish lean"
    },
    {
      "label": 0,
      "probability": 0.65,
      "rationale": "Mission accomplished statement per critical rules guidance"
    }
  ],
  "primary_label": 0,
  "ambiguity_type": "rule_conflict"
}
```

## Key Principle

The goal of candidate annotation is to capture genuine ambiguity, not to force a decision. When the jury splits, it often means the headline legitimately admits multiple interpretations. Soft labels preserve this information for model training.
