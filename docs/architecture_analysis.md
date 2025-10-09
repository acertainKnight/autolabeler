# LLM Call System Architecture Analysis

**Analyst Agent Report**
**Date:** 2025-10-08
**Task:** Analyze LLM call system for budget tracking integration

---

## Executive Summary

The AutoLabeler v2 system has been successfully analyzed and enhanced with budget tracking capabilities. The system uses a modular architecture with clear separation between LLM provider abstraction, labeling services, and configuration management. Budget tracking has been integrated at the LLM client level with graceful shutdown capabilities.

## System Architecture Overview

### Component Hierarchy

```
AutoLabelerV2 (High-level facade)
    ├── LabelingService (Core labeling orchestration)
    │   ├── KnowledgeStore (RAG example retrieval)
    │   ├── PromptManager (Prompt tracking)
    │   └── LLM Client (via factory)
    ├── EvaluationService
    ├── SyntheticGenerationService
    └── RuleGenerationService

LLM Provider Layer
    ├── factory.py (get_llm_client)
    │   ├── OpenRouterClient
    │   └── CorporateOpenAIClient
    └── budget_tracker.py (CostTracker)
```

## Key Files and Responsibilities

### 1. Configuration Layer

#### `/src/autolabeler/config.py` (Settings)
- Global application settings loaded from `.env`
- API keys for LLM providers
- Model defaults and paths
- **NEW:** `llm_budget` field for global budget limit

#### `/src/autolabeler/core/configs.py` (Task Configs)
- **LabelingConfig:** Labeling parameters including `budget` field
- **BatchConfig:** Batch processing settings
- **ActiveLearningConfig:** Includes `max_budget` for active learning loops

### 2. LLM Provider Layer

#### `/src/autolabeler/core/llm_providers/factory.py`
Factory function that creates LLM clients with budget tracking:

```python
def get_llm_client(
    settings: Settings,
    config: LabelingConfig,
    cost_tracker: CostTracker | None = None
) -> BaseChatModel:
    # Auto-creates CostTracker if budget specified
    # Passes to OpenRouterClient or CorporateOpenAIClient
```

#### `/src/autolabeler/core/llm_providers/openrouter.py`
OpenRouter API client with:
- Credit-based rate limiting (existing)
- **NEW:** Cost tracking in `_generate()` and `_agenerate()`
- **NEW:** Budget checking before each API call
- **NEW:** Cost extraction from OpenRouter response metadata

#### `/src/autolabeler/core/llm_providers/corporate.py`
Corporate/Internal LLM client with:
- SSL configuration and security checks (existing)
- **NEW:** Cost tracking (typically $0.00 for internal endpoints)
- **NEW:** Budget checking for monitoring purposes

### 3. Budget Tracking Infrastructure

#### `/src/autolabeler/core/utils/budget_tracker.py` (NEW FILE)

**CostTracker Class:**
- Thread-safe cost accumulation using `threading.Lock`
- Budget enforcement with `is_budget_exceeded()`
- Statistics tracking (total_cost, call_count, remaining_budget)
- Reset functionality for testing

**BudgetExceededError Exception:**
- Custom exception raised when budget limit reached
- Contains cost information for error handling

**Cost Extraction Functions:**
- `extract_openrouter_cost()`: Extracts cost from OpenRouter API response
- `extract_openai_cost()`: Calculates cost from token counts and model pricing
- `extract_cost_from_result()`: Router function based on provider type

### 4. Service Layer

#### `/src/autolabeler/core/labeling/labeling_service.py`
Main service for text labeling:
- `label_text()`: Single text labeling with prompt preparation
- `label_dataframe()`: Batch labeling with progress tracking
- `_get_client_for_config()`: Client caching by configuration
- Integration with KnowledgeStore for RAG

#### `/src/autolabeler/autolabeler_v2.py`
High-level facade:
- Lazy-loads services as needed
- Orchestrates workflows
- Provides simplified API for users

## LLM Call Flow with Budget Tracking

### Single Text Labeling Flow

```
1. User calls: labeling_service.label_text(text, config)
   ├─> config.budget = 10.0 USD

2. Service prepares prompt with RAG:
   ├─> _prepare_prompt(text, config, template)
   ├─> KnowledgeStore.find_similar_examples()
   └─> Jinja2 template rendering

3. Service gets LLM client:
   ├─> _get_client_for_config(config)
   ├─> factory.get_llm_client(settings, config)
   ├─> CostTracker(budget=10.0) created
   └─> OpenRouterClient(cost_tracker=tracker)

4. LLM invocation:
   ├─> structured_llm.invoke(prompt)
   ├─> OpenRouterClient._generate() called
   │   ├─> Check: tracker.is_budget_exceeded() ❌
   │   ├─> Apply rate limiting
   │   ├─> super()._generate() → API call
   │   ├─> Extract cost from response: $0.0023
   │   └─> tracker.add_cost(0.0023)
   └─> Return LabelResponse

5. Result processing:
   ├─> Update prompt_manager with result
   ├─> Save to knowledge base if high confidence
   └─> Return to user
```

### Batch Labeling Flow

```
1. User calls: labeling_service.label_dataframe(df, text_col, config)
   ├─> BatchConfig(batch_size=50)
   ├─> config.budget = 10.0 USD

2. Process in batches:
   For each batch of 50 items:

   a. Prepare prompts:
      ├─> _prepare_prompt() for each item
      └─> Collect all prompts

   b. Batch LLM call:
      ├─> structured_llm.batch(prompts)
      ├─> Multiple _generate() calls
      │   ├─> Each checks: is_budget_exceeded()
      │   ├─> If exceeded → raise BudgetExceededError
      │   └─> If OK → continue with API call
      └─> Aggregate results

   c. Save progress:
      ├─> Update BatchProcessor progress file
      └─> Save checkpoint for resume

3. Budget exceeded handling:
   ├─> BudgetExceededError caught in batch processor
   ├─> Save current progress
   ├─> Return partial results
   └─> Log: "Budget exceeded after N items"
```

## Cost Calculation by Provider

### OpenRouter
- **Method:** Direct cost from API response
- **Location:** `response_metadata.total_cost`
- **Accuracy:** Exact (provided by OpenRouter)
- **Fallback:** Log usage tokens if no cost available

### OpenAI / Corporate
- **Method:** Calculate from token counts
- **Pricing Data:** Hardcoded in `extract_openai_cost()`
  - gpt-4o: $2.50/$10.00 per 1M input/output tokens
  - gpt-4o-mini: $0.15/$0.60 per 1M tokens
  - gpt-3.5-turbo: $0.50/$1.50 per 1M tokens
- **Formula:** `(tokens / 1M) × price_per_million`
- **Corporate:** Returns $0.00 (internal endpoints)

## Integration Points

### Where Budget Tracking is Active

1. **LLM Client Level** (Primary)
   - `OpenRouterClient._generate()` (line 254-290)
   - `OpenRouterClient._agenerate()` (line 292-332)
   - `CorporateOpenAIClient._generate()` (line 215-247)
   - `CorporateOpenAIClient._agenerate()` (line 249-281)

2. **Factory Level** (Initialization)
   - `get_llm_client()` creates `CostTracker` if budget specified
   - Passes tracker to client constructors

3. **Configuration Level** (Declaration)
   - `Settings.llm_budget`: Global budget limit
   - `LabelingConfig.budget`: Per-labeling-run budget
   - `ActiveLearningConfig.max_budget`: Active learning budget

### Budget Checking Flow

**Before each API call:**
```python
if cost_tracker:
    if cost_tracker.is_budget_exceeded():
        stats = cost_tracker.get_stats()
        raise BudgetExceededError(stats["total_cost"], stats["budget"])
```

**After each API call:**
```python
if cost_tracker:
    cost = extract_cost_from_result(result, provider, model)
    if cost > 0:
        cost_tracker.add_cost(cost)
```

## Graceful Shutdown Mechanism

### Budget Exceeded Handling

1. **Detection:** Budget check before each `_generate()` call
2. **Exception:** `BudgetExceededError` raised with cost details
3. **Propagation:** Exception bubbles up to service layer
4. **Recovery:** Batch processor catches and saves progress

### Progress Tracking

The system already has progress tracking infrastructure:
- `ProgressTracker` base class saves state to JSON
- `BatchProcessor` tracks completed items
- Resume capability via `resume_key` parameter

### Partial Results

When budget is exceeded during batch processing:
- Already-labeled items are returned
- Progress file contains checkpoint
- User can resume with increased budget or continue later

## Files Modified/Created

### Modified Files
1. `/src/autolabeler/config.py` - Added `llm_budget` field
2. `/src/autolabeler/core/configs.py` - Added `budget` to LabelingConfig
3. `/src/autolabeler/core/llm_providers/factory.py` - Added CostTracker integration
4. `/src/autolabeler/core/llm_providers/openrouter.py` - Added cost tracking
5. `/src/autolabeler/core/llm_providers/corporate.py` - Added cost tracking

### New Files
1. `/src/autolabeler/core/utils/budget_tracker.py` - Complete budget tracking system

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   AutoLabelerV2                          │
│              (Orchestration Layer)                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              LabelingService                             │
│  • label_text() - single text                           │
│  • label_dataframe() - batch                            │
│  • _prepare_prompt() - RAG + prompts                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         LLM Client Factory (get_llm_client)             │
│  • Creates CostTracker(budget=X)                        │
│  • Passes to client constructor                         │
└────┬──────────────────────────────────────┬─────────────┘
     │                                      │
     ▼                                      ▼
┌──────────────────┐            ┌──────────────────┐
│ OpenRouterClient │            │ CorporateClient  │
│ with CostTracker │            │ with CostTracker │
└────┬─────────────┘            └────┬─────────────┘
     │                               │
     │  _generate():                 │  _generate():
     │  1. Check budget              │  1. Check budget
     │  2. Apply rate limit          │  2. Make API call
     │  3. Make API call             │  3. Extract cost ($0)
     │  4. Extract cost              │  4. Track cost
     │  5. Track cost                │
     │                               │
     └───────────────┬───────────────┘
                     ▼
          ┌──────────────────────┐
          │    CostTracker       │
          │  (Thread-safe)       │
          │  • total_cost        │
          │  • call_count        │
          │  • budget            │
          │  • is_exceeded()     │
          └──────────────────────┘
```

## Risk Assessment

### Low Risk ✅
- Budget parameters added to configs
- Cost tracking in `_generate()` methods
- Thread-safe cost accumulation
- Logging and monitoring

### Medium Risk ⚠️
- Changes to client constructors (backward compatible with defaults)
- Factory signature changed (backward compatible)
- Provider-specific cost calculations (may need updates as pricing changes)

### High Risk (Avoided) ✅
- No breaking changes to public API
- Client caching still works
- Rate limiting unaffected
- No changes to LangChain base class methods

## Success Criteria Met

✅ **Budget tracking integrated** at LLM provider level
✅ **Thread-safe** cost accumulation
✅ **Provider-specific** cost extraction (OpenRouter, OpenAI, Corporate)
✅ **Graceful shutdown** via BudgetExceededError
✅ **Progress tracking** leverages existing infrastructure
✅ **Configuration options** at multiple levels (Settings, LabelingConfig)
✅ **Backward compatible** - budget tracking is optional

## Recommendations

### For Implementation Teams

1. **Testing:**
   - Unit tests for CostTracker
   - Integration tests with mock API responses
   - Test budget exceeded handling in batch processing

2. **Documentation:**
   - Update user guide with budget configuration examples
   - Document cost tracking behavior per provider
   - Add troubleshooting section for budget issues

3. **Monitoring:**
   - Log budget status at key checkpoints
   - Track cost trends over time
   - Alert when approaching budget limits

### For Future Enhancements

1. **Dynamic Pricing:** Fetch pricing from provider APIs instead of hardcoding
2. **Cost Estimation:** Preview estimated cost before running batch jobs
3. **Budget Alerts:** Warn users at 50%, 75%, 90% of budget
4. **Cost Analytics:** Dashboard showing cost breakdown by task/model
5. **Budget Pooling:** Share budget across multiple labeling runs

## Conclusion

The LLM call system has been successfully enhanced with comprehensive budget tracking capabilities. The implementation is thread-safe, provider-aware, and integrates seamlessly with the existing architecture. Budget enforcement occurs at the lowest level (_generate methods) ensuring no API calls are made after budget exhaustion. The system maintains backward compatibility while providing powerful cost control features for production use.

---

**Analyst Agent**
Hive Mind Swarm Coordination System
