# Budget Tracking Integration Summary

## Quick Reference

### Files Modified ✏️

| File | Changes | Purpose |
|------|---------|---------|
| `config.py` | Added `llm_budget: float \| None` | Global budget limit |
| `core/configs.py` | Added `budget: float \| None` to LabelingConfig | Per-run budget limit |
| `llm_providers/factory.py` | Added `cost_tracker` parameter, auto-creates from budget | Client initialization |
| `llm_providers/openrouter.py` | Cost tracking in `_generate()` and `_agenerate()` | Real-time cost tracking |
| `llm_providers/corporate.py` | Cost tracking in `_generate()` and `_agenerate()` | Real-time cost tracking |

### New File Created 🆕

**`core/utils/budget_tracker.py`** (298 lines)
- `CostTracker`: Thread-safe cost accumulation
- `BudgetExceededError`: Budget limit exception
- Cost extraction functions for each provider

## Integration Points

### 1️⃣ Configuration Level
```python
# Global setting
settings = Settings(llm_budget=50.0)  # $50 total budget

# Per-labeling-run
config = LabelingConfig(budget=10.0)  # $10 for this run
```

### 2️⃣ Factory Level
```python
# Automatic tracker creation
client = get_llm_client(
    settings=settings,
    config=config,
    cost_tracker=None  # Auto-created from config.budget or settings.llm_budget
)
```

### 3️⃣ Client Level
```python
class OpenRouterClient:
    def _generate(self, *args, **kwargs):
        # 1. Check budget BEFORE API call
        if self._cost_tracker.is_budget_exceeded():
            raise BudgetExceededError(total_cost, budget)

        # 2. Make API call
        result = super()._generate(*args, **kwargs)

        # 3. Track cost AFTER API call
        cost = extract_cost_from_result(result, "openrouter", model)
        self._cost_tracker.add_cost(cost)

        return result
```

## Cost Extraction Methods

### OpenRouter
- **Source:** `response_metadata.total_cost` from API response
- **Type:** Direct (exact cost provided by API)
- **Fallback:** Logs token usage if cost unavailable

### OpenAI/Corporate
- **Source:** `token_usage` from API response
- **Type:** Calculated using hardcoded pricing
- **Formula:** `(prompt_tokens / 1M × input_price) + (completion_tokens / 1M × output_price)`

## Budget Enforcement Flow

```
User Request
    │
    ▼
[LabelingService]
    │
    ├─> Config: budget = $10.00
    │
    ▼
[Factory: get_llm_client]
    │
    ├─> Create CostTracker(budget=10.00)
    │
    ▼
[OpenRouterClient]
    │
    ├─> Store: self._cost_tracker = tracker
    │
    ▼
[API Call Loop]
    │
    ├─> _generate() called
    │   │
    │   ├─> CHECK: is_budget_exceeded()?
    │   │   ├─> NO  → Continue
    │   │   └─> YES → raise BudgetExceededError ❌
    │   │
    │   ├─> Make API call
    │   │
    │   └─> Extract & track cost: $0.0023
    │       └─> total_cost += $0.0023
    │
    ▼
[Result or Exception]
```

## Example Usage

### Basic Usage
```python
from autolabeler import AutoLabelerV2, Settings, LabelingConfig

# Set budget
settings = Settings(openrouter_api_key="sk-...", llm_budget=25.0)
config = LabelingConfig(budget=10.0)  # Overrides global

# Create labeler
labeler = AutoLabelerV2("sentiment", settings, config)

# Label with budget enforcement
try:
    results = labeler.label(df, "text")
except BudgetExceededError as e:
    print(f"Budget exceeded: ${e.total_cost:.2f} / ${e.budget:.2f}")
    # Partial results are still available
```

### Checking Budget Status
```python
# Get cost tracker from client
client = labeler.labeling_service._get_client_for_config(config)
tracker = client._instance_data[id(client)].cost_tracker

# Check stats
stats = tracker.get_stats()
print(f"Total cost: ${stats['total_cost']:.4f}")
print(f"Remaining: ${stats['remaining_budget']:.4f}")
print(f"API calls: {stats['call_count']}")
```

## Graceful Shutdown

When budget is exceeded during batch processing:

1. **Detection:** Budget check before each API call
2. **Exception:** `BudgetExceededError` raised
3. **Progress Saved:** Batch processor saves checkpoint
4. **Partial Results:** Already-labeled items returned
5. **Resume:** User can resume with increased budget

## Thread Safety

The `CostTracker` uses `threading.Lock` for thread-safe operations:

```python
@dataclass
class CostTracker:
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add_cost(self, cost: float) -> bool:
        with self._lock:
            self.total_cost += cost
            self.call_count += 1
            # Check budget...
```

## Cost Estimation (Not Yet Implemented)

Future enhancement ideas:
- Preview estimated cost before running batch jobs
- Based on average tokens per example
- Alert at 50%, 75%, 90% of budget
- Dynamic pricing from provider APIs

## Pricing Reference (as of 2025-01)

### OpenAI Models
| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|----------------------|
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-3.5-turbo | $0.50 | $1.50 |

### OpenRouter
- Varies by model
- Cost provided directly in API response
- Check: https://openrouter.ai/models

### Corporate/Internal
- Typically $0.00 (free internal endpoints)
- Cost tracking available for monitoring

## Testing Checklist

- [ ] Unit tests for `CostTracker`
  - [ ] Thread-safe cost accumulation
  - [ ] Budget exceeded detection
  - [ ] Statistics reporting

- [ ] Unit tests for cost extraction
  - [ ] OpenRouter cost extraction
  - [ ] OpenAI cost calculation
  - [ ] Corporate zero-cost handling

- [ ] Integration tests
  - [ ] Budget enforcement during labeling
  - [ ] BudgetExceededError handling
  - [ ] Progress saving on budget exceeded
  - [ ] Partial results return

- [ ] E2E tests
  - [ ] Batch labeling with budget
  - [ ] Resume after budget exceeded
  - [ ] Multiple concurrent labeling runs

## Troubleshooting

### Budget exceeded immediately
- Check if budget is too low for even one API call
- Verify cost extraction is working correctly
- Check logs for cost per call

### Cost not being tracked
- Verify `cost_tracker` is passed to client
- Check if provider cost extraction is implemented
- Enable debug logging to see cost extraction details

### Inconsistent cost tracking
- Ensure using same CostTracker instance across calls
- Check for race conditions in multithreaded scenarios
- Verify thread-safety with lock debugging

---

**For detailed architecture analysis, see:** `/docs/architecture_analysis.md`
