# Candidate Annotation Prompt (for Jury Disagreements)

The jury disagreed on this article. Instead of forcing a single label, list ALL PLAUSIBLE labels with your estimated probability for each.

## Task

Analyze why this article is ambiguous with respect to trade policy uncertainty and which criteria might support different interpretations. Then output ALL labels that have reasonable support, with probability weights that sum to 1.0.

## Response Format

Return ONLY a JSON object with this structure:

```json
{
  "reasoning": "Brief analysis of why this is ambiguous and which criteria apply to each interpretation",
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
  "ambiguity_type": "causal_link_unclear | scope_boundary | criteria_conflict"
}
```

## Requirements

- Probabilities must sum to 1.0
- Only include labels with probability ≥ 0.15
- Most ambiguous cases will have 2 candidates (e.g., {0: 0.6, 1: 0.4})
- For truly split cases, probabilities can be close (e.g., {0: 0.5, 1: 0.5})

## Ambiguity Types

- **causal_link_unclear**: Trade policy and uncertainty are both mentioned but their causal connection is debatable
- **scope_boundary**: Ambiguity about whether the topic qualifies as "trade policy" vs adjacent concepts (e.g., trade volumes, domestic regulation)
- **criteria_conflict**: Different classification criteria support different labels

## Common Sources of Ambiguity

1. **Causal link gap**: Article mentions both trade policy and uncertainty, but the uncertainty may stem from a different cause (e.g., inflation, elections)
2. **Implemented but contested**: A policy is in effect but under review or debate — does this count as uncertainty?
3. **Indirect trade policy reference**: Article discusses economic consequences of trade policy without naming specific policies
4. **Risk vs certainty**: Article discusses trade policy risks in a way that could be read as certain prediction rather than uncertainty
5. **Scope edge cases**: Topics like currency manipulation, sanctions, or industrial policy that overlap with trade policy

## Examples

### Example 1: Causal Link Unclear (0 vs 1)
Text: "Manufacturing output fell sharply amid global trade disruptions and supply chain challenges."

```json
{
  "reasoning": "'Trade disruptions' could imply trade policy uncertainty, but 'supply chain challenges' may be operational. The causal link between policy uncertainty and the output decline is ambiguous.",
  "candidates": [
    {
      "label": 0,
      "probability": 0.55,
      "rationale": "Trade disruptions described here appear to be operational/logistical rather than policy-driven"
    },
    {
      "label": 1,
      "probability": 0.45,
      "rationale": "'Trade disruptions' could encompass policy-driven disruptions like tariff changes"
    }
  ],
  "primary_label": 0,
  "ambiguity_type": "causal_link_unclear"
}
```

### Example 2: Scope Boundary (0 vs 1)
Text: "The government announced new sanctions on technology exports, leaving semiconductor firms uncertain about future sales."

```json
{
  "reasoning": "Export sanctions are trade-adjacent policy. 'Uncertain about future sales' is clear uncertainty language. The question is whether sanctions on technology exports qualify as trade policy.",
  "candidates": [
    {
      "label": 1,
      "probability": 0.70,
      "rationale": "Export sanctions directly restrict international trade and firms express uncertainty about their impact"
    },
    {
      "label": 0,
      "probability": 0.30,
      "rationale": "Sanctions may be classified as national security policy rather than trade policy per se"
    }
  ],
  "primary_label": 1,
  "ambiguity_type": "scope_boundary"
}
```

## Key Principle

The goal of candidate annotation is to capture genuine ambiguity, not to force a decision. When the jury splits, it often means the article legitimately admits multiple interpretations. Soft labels preserve this information for model training.
