# Federal Reserve Headlines Labeling Pipeline

## Overview

This pipeline labels Federal Reserve headline data with multiple classification tasks using a two-phase approach:
1. **Phase 1** (1000 examples): Learn optimal rules through active learning
2. **Phase 2** (19,000 examples): Apply learned rules to remaining data

Total output: 20,000 labeled headlines sampled uniformly across time

## Classification Tasks (11 Total)

### 1. Relevancy (2 labels)
- **relevant**: Headline discusses US monetary policy or economy AND spoken by Fed member
- **not_relevant**: About regulations, crypto, other countries, or non-Fed speakers

### 2. Hawk/Dove Scale (5 labels: -2 to 2)
- **-2** (Explicitly Dovish): Direct calls for rate cuts or easing ("rates should be lowered")
- **-1** (Implicitly Dovish): Economic weakness suggesting need for cuts (slowing inflation, weak labor market)
- **0** (Neutral): No directional signal, balanced risks, "wait and see"
- **+1** (Implicitly Hawkish): Economic strength suggesting need for hikes (high inflation, tight labor)
- **+2** (Explicitly Hawkish): Direct calls for rate hikes or tightening ("rates should rise")

### 3. Speaker (7 labels)
- fed_chair, fed_vice_chair, fed_governor, regional_president, other_fed, non_fed, unknown

### 4-11. Topic Binary Indicators (yes/no for each)
- **inflation**: Price increases, inflation expectations, PCE, CPI
- **labor_market**: Employment, unemployment, wages, job growth
- **growth**: GDP, expansion, recession, economic activity
- **international**: Foreign economies, trade, exchange rates (relevant to US policy)
- **housing**: Housing market, home prices, mortgage rates
- **consumer_spending**: Consumption, retail sales, consumer confidence
- **financial_markets**: Financial conditions, credit markets, market functioning

## Files Created

### 1. Task Configuration: `configs/fed_headlines_tasks.json`
**Purpose**: Defines all 11 classification tasks with initial principles

**Structure**:
```json
{
  "task_name": {
    "labels": ["label1", "label2", ...],
    "principles": [
      "Principle 1: Clear classification rule",
      "Principle 2: Examples and edge cases",
      ...
    ]
  }
}
```

**Key Principles Encoded**:
- Relevancy requires BOTH monetary policy/economy AND Fed speaker
- Hawk/dove scale with explicit vs implicit distinctions
- Topics must be central to statement (not passing mentions)
- Speaker identification from headline text

### 2. Main Pipeline: `scripts/fed_headlines_labeling_pipeline.py`
**Purpose**: Complete two-phase labeling implementation

**Key Functions**:

#### `sample_data_across_time(df, n_samples, date_column)`
- Samples data uniformly across time period
- Ensures representative coverage of entire dataset
- Sorts by capturetime and takes evenly-spaced indices

#### `phase1_learn_rules(input_file, output_file, task_configs_file, ...)`
**What it does**:
1. Samples 1000 headlines across time
2. Processes in 50-row batches
3. After each batch: identifies uncertain predictions (confidence < 0.7)
4. Generates improved rules via LLM based on error patterns
5. Updates principles for next batch
6. Returns learned rules dictionary

**Outputs**:
- `outputs/fed_headlines/phase1_labeled_1000.csv` - Labeled data
- Learned rules dictionary (returned, then saved as JSON)

**Statistics logged**:
- Total error patterns identified
- Rules generated count
- Pattern types distribution
- Confidence statistics per task

#### `phase2_full_labeling(input_file, output_file, task_configs_file, learned_rules, ...)`
**What it does**:
1. Samples 20,000 headlines across time
2. Skips first 1000 (already labeled in Phase 1)
3. Processes remaining 19,000 in 100-row batches
4. Uses learned rules from Phase 1 (NO rule evolution)
5. Faster processing with larger batches

**Outputs**:
- `outputs/fed_headlines/phase2_labeled_19000.csv` - Labeled data

#### `run_complete_pipeline(...)`
**What it does**:
1. Runs Phase 1 with rule learning
2. Saves learned rules to JSON
3. Runs Phase 2 with learned rules
4. Combines Phase 1 + Phase 2 into final dataset

**Outputs**:
- `outputs/fed_headlines/phase1_labeled_1000.csv`
- `outputs/fed_headlines/phase1_learned_rules.json`
- `outputs/fed_headlines/phase2_labeled_19000.csv`
- `outputs/fed_headlines/fed_headlines_labeled_20000.csv` (combined)

### 3. Manual Execution Script: `scripts/fed_headlines_manual_run.sh`
**Purpose**: Convenient shell interface for running pipeline

**Commands**:
```bash
# Run Phase 1 only (1000 examples, learn rules)
./scripts/fed_headlines_manual_run.sh phase1

# Run Phase 2 only (19000 examples, requires Phase 1 complete)
./scripts/fed_headlines_manual_run.sh phase2

# Run complete pipeline (both phases)
./scripts/fed_headlines_manual_run.sh complete
```

**Safety checks**:
- Verifies config and data files exist
- Creates output directory automatically
- Validates Phase 1 completion before Phase 2
- Clear error messages

## Output Format

Each labeled row contains original columns plus:

For each task (11 tasks × 3 columns = 33 new columns):
- `label_{task_name}`: Predicted label
- `confidence_{task_name}`: Confidence score (0.0-1.0)
- `reasoning_{task_name}`: Explanation of classification

**Example columns**:
```
headline, capturetime, ...original columns...,
label_relevancy, confidence_relevancy, reasoning_relevancy,
label_hawk_dove, confidence_hawk_dove, reasoning_hawk_dove,
label_speaker, confidence_speaker, reasoning_speaker,
label_topic_inflation, confidence_topic_inflation, reasoning_topic_inflation,
...
```

## Running the Pipeline

### Quick Start (Complete Pipeline)
```bash
# Option 1: Run Python script directly
python3 scripts/fed_headlines_labeling_pipeline.py

# Option 2: Use shell wrapper
./scripts/fed_headlines_manual_run.sh complete
```

### Manual Phase-by-Phase Execution
```bash
# Step 1: Run Phase 1 (learn rules)
./scripts/fed_headlines_manual_run.sh phase1
# Review: outputs/fed_headlines/phase1_learned_rules.json

# Step 2: Run Phase 2 (apply learned rules)
./scripts/fed_headlines_manual_run.sh phase2
```

### Using CLI Directly (Alternative)
```bash
autolabeler label-multi \
    --dataset-name "fed_headlines" \
    --input-file datasets/fed_data_full.csv \
    --output-file outputs/fed_headlines_labeled.csv \
    --text-column headline \
    --tasks "relevancy,hawk_dove,speaker,topic_inflation,topic_labor_market,topic_growth,topic_international,topic_housing,topic_consumer_spending,topic_financial_markets" \
    --task-configs configs/fed_headlines_tasks.json \
    --enable-rule-evolution \
    --batch-size 50 \
    --confidence-threshold 0.7
```

## Expected Runtime

**Phase 1** (1000 examples, 50/batch):
- ~20 batches with rule evolution
- ~5-10 minutes per batch (depends on LLM speed)
- **Total: ~2-3 hours**

**Phase 2** (19,000 examples, 100/batch):
- ~190 batches without rule evolution
- ~2-5 minutes per batch
- **Total: ~6-15 hours**

**Complete pipeline**: ~8-18 hours total

**Optimization**: Run overnight or use higher batch sizes

## Configuration Options

### In `fed_headlines_labeling_pipeline.py`:

```python
run_complete_pipeline(
    input_file="datasets/fed_data_full.csv",
    output_dir="outputs/fed_headlines",
    task_configs_file="configs/fed_headlines_tasks.json",
    phase1_samples=1000,           # Samples for rule learning
    phase2_total=20000,            # Total samples (includes Phase 1)
    phase1_batch_size=50,          # Smaller batches for rule evolution
    phase2_batch_size=100,         # Larger batches for efficiency
    confidence_threshold=0.7       # Threshold for uncertain predictions
)
```

### In Settings (Environment Variables or Code):

```python
from autolabeler.config import Settings

settings = Settings()
settings.llm_model = "claude-3-5-sonnet-20241022"  # or gpt-4
settings.llm_provider = "anthropic"  # or "openai"
settings.temperature = 0.1  # Low for consistency
```

## Reviewing Results

### Check Phase 1 Output
```python
import pandas as pd

# Load Phase 1 results
df = pd.read_csv("outputs/fed_headlines/phase1_labeled_1000.csv")

# Check confidence distributions
for task in ["relevancy", "hawk_dove", "speaker"]:
    print(f"\n{task}:")
    print(f"  Mean confidence: {df[f'confidence_{task}'].mean():.3f}")
    print(f"  Label distribution:")
    print(df[f'label_{task}'].value_counts())

# Check learned rules
import json
with open("outputs/fed_headlines/phase1_learned_rules.json") as f:
    rules = json.load(f)

print("\nLearned rules for relevancy:")
for i, rule in enumerate(rules["relevancy"], 1):
    print(f"  {i}. {rule}")
```

### Check Phase 2 Output
```python
# Load Phase 2 results
df2 = pd.read_csv("outputs/fed_headlines/phase2_labeled_19000.csv")

# Compare confidence between phases
df1 = pd.read_csv("outputs/fed_headlines/phase1_labeled_1000.csv")

for task in ["relevancy", "hawk_dove"]:
    conf1 = df1[f"confidence_{task}"].mean()
    conf2 = df2[f"confidence_{task}"].mean()
    print(f"{task}: Phase 1={conf1:.3f}, Phase 2={conf2:.3f}, Δ={conf2-conf1:+.3f}")
```

## Troubleshooting

### Low Confidence Scores
- Review learned rules in `phase1_learned_rules.json`
- Check if initial principles in task config are clear enough
- Increase `phase1_samples` for more rule learning
- Lower `confidence_threshold` to reduce uncertainty detection

### Incorrect Classifications
- Review reasoning in `reasoning_{task}` columns
- Refine principles in `configs/fed_headlines_tasks.json`
- Add more specific examples to principles
- Re-run Phase 1 with adjusted initial principles

### Out of Memory
- Reduce `phase1_batch_size` and `phase2_batch_size`
- Process in smaller chunks
- Use less powerful but smaller LLM model

### API Rate Limits
- Reduce batch size
- Add delays between batches (modify pipeline script)
- Use different API key with higher limits

## Advanced Customization

### Modify Task Definitions
Edit `configs/fed_headlines_tasks.json`:
```json
{
  "your_task": {
    "labels": ["label1", "label2"],
    "principles": [
      "Clear rule for classification",
      "Examples and edge cases"
    ]
  }
}
```

### Change Sampling Strategy
Edit `sample_data_across_time()` in pipeline script:
```python
# Random sampling instead of time-based
sampled_df = df.sample(n=n_samples, random_state=42)

# Stratified sampling by year
sampled_df = df.groupby(df['capturetime'].dt.year).sample(
    n=n_samples_per_year
)
```

### Custom Rule Evolution Strategy
```python
rule_evolution = RuleEvolutionService(
    initial_rules=initial_principles,
    improvement_strategy="heuristic",  # Instead of "feedback_driven"
    settings=settings
)
```

## Next Steps After Labeling

1. **Quality Check**: Review sample of labeled data manually
2. **Error Analysis**: Identify systematic labeling errors
3. **Model Training**: Use labels as training data for faster models
4. **Validation**: Compare against manually labeled gold standard
5. **Iteration**: Refine principles and re-run if needed

## File Structure Summary

```
autolabeler/
├── configs/
│   └── fed_headlines_tasks.json        # Task definitions & principles
├── scripts/
│   ├── fed_headlines_labeling_pipeline.py  # Main pipeline code
│   └── fed_headlines_manual_run.sh         # Shell wrapper
├── datasets/
│   └── fed_data_full.csv               # Input data (100k rows)
├── outputs/
│   └── fed_headlines/
│       ├── phase1_labeled_1000.csv      # Phase 1 output
│       ├── phase1_learned_rules.json    # Learned rules
│       ├── phase2_labeled_19000.csv     # Phase 2 output
│       └── fed_headlines_labeled_20000.csv  # Combined final output
└── docs/
    └── fed-headlines-labeling-guide.md  # This file
```
