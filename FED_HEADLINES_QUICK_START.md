# Fed Headlines Labeling - Quick Start Guide

## What Was Set Up

A complete two-phase multi-label classification pipeline for 100,000 Federal Reserve headlines:

**Phase 1**: 1,000 examples → Learn optimal classification rules
**Phase 2**: 19,000 examples → Apply learned rules
**Output**: 20,000 labeled headlines across 11 classification tasks

## Classification Tasks (11 Total)

1. **relevancy**: relevant | not_relevant
2. **hawk_dove**: -2 (dovish) to +2 (hawkish)
3. **speaker**: fed_chair | fed_vice_chair | fed_governor | regional_president | other_fed | non_fed | unknown
4. **topic_inflation**: yes | no
5. **topic_labor_market**: yes | no
6. **topic_growth**: yes | no
7. **topic_international**: yes | no
8. **topic_housing**: yes | no
9. **topic_consumer_spending**: yes | no
10. **topic_financial_markets**: yes | no

## Files Created

### 1. Task Configuration
**Location**: `configs/fed_headlines_tasks.json`
**Contains**: All 11 task definitions with initial classification principles

### 2. Main Pipeline
**Location**: `scripts/fed_headlines_labeling_pipeline.py`
**Contains**: Complete two-phase implementation with rule learning

### 3. Shell Script
**Location**: `scripts/fed_headlines_manual_run.sh`
**Contains**: Convenient commands for running pipeline phases

### 4. Documentation
**Location**: `docs/fed-headlines-labeling-guide.md`
**Contains**: Comprehensive guide with examples and troubleshooting

## How to Run

### Option 1: Complete Pipeline (Recommended)
```bash
# Run both phases automatically
python3 scripts/fed_headlines_labeling_pipeline.py
```

### Option 2: Manual Phase Control
```bash
# Phase 1: Learn rules from 1000 examples
./scripts/fed_headlines_manual_run.sh phase1

# Review learned rules
cat outputs/fed_headlines/phase1_learned_rules.json

# Phase 2: Label remaining 19000 examples
./scripts/fed_headlines_manual_run.sh phase2
```

### Option 3: Using Shell Wrapper
```bash
# Complete pipeline
./scripts/fed_headlines_manual_run.sh complete
```

## Output Files

All outputs saved to: `outputs/fed_headlines/`

1. **phase1_labeled_1000.csv** - First 1000 labeled examples
2. **phase1_learned_rules.json** - Rules learned from Phase 1
3. **phase2_labeled_19000.csv** - Remaining 19000 labeled examples
4. **fed_headlines_labeled_20000.csv** - Combined final dataset

## Expected Runtime

- **Phase 1**: ~2-3 hours (1000 examples with rule learning)
- **Phase 2**: ~6-15 hours (19000 examples, no rule learning)
- **Total**: ~8-18 hours

**Tip**: Run overnight or in background

## Output Format

Each row contains 33 new columns (11 tasks × 3):
- `label_{task}` - Predicted label
- `confidence_{task}` - Confidence score (0-1)
- `reasoning_{task}` - Explanation

Example:
```
headline,capturetime,...,
label_relevancy,confidence_relevancy,reasoning_relevancy,
label_hawk_dove,confidence_hawk_dove,reasoning_hawk_dove,
...
```

## Key Features

✅ **Time-based sampling**: Uniformly samples across entire dataset timeline
✅ **Rule evolution**: Learns optimal principles from uncertain predictions
✅ **Single LLM call**: All 11 labels determined simultaneously (efficient)
✅ **Confidence tracking**: Identifies uncertain predictions for review
✅ **Comprehensive logging**: Progress and statistics at every step

## Quick Validation

After Phase 1 completes, check results:
```python
import pandas as pd

df = pd.read_csv("outputs/fed_headlines/phase1_labeled_1000.csv")

# Check label distributions
print(df['label_relevancy'].value_counts())
print(df['label_hawk_dove'].value_counts())

# Check mean confidences
for task in ['relevancy', 'hawk_dove', 'speaker']:
    print(f"{task}: {df[f'confidence_{task}'].mean():.3f}")
```

## Configuration

Edit `configs/fed_headlines_tasks.json` to:
- Modify classification principles
- Add more examples to rules
- Adjust label definitions

Edit `scripts/fed_headlines_labeling_pipeline.py` to:
- Change batch sizes (phase1_batch_size, phase2_batch_size)
- Adjust sample counts (phase1_samples, phase2_total)
- Modify confidence threshold

## Troubleshooting

**Low confidence**: Review learned rules, refine initial principles
**Wrong labels**: Check reasoning columns, adjust task principles
**Out of memory**: Reduce batch sizes
**API limits**: Reduce batch size, add delays

## Next Steps

1. **Review** Phase 1 results and learned rules
2. **Run** Phase 2 if Phase 1 looks good
3. **Validate** sample of final labels manually
4. **Iterate** if needed by adjusting principles

## Support

- Full documentation: `docs/fed-headlines-labeling-guide.md`
- Pipeline code: `scripts/fed_headlines_labeling_pipeline.py`
- Task config: `configs/fed_headlines_tasks.json`
