# Cost Tracking API Research Report

**Date:** 2025-10-08
**Researcher:** Research Agent (Hive Mind Swarm)
**Task:** Investigate cost tracking mechanisms for OpenRouter, OpenAI, and Corporate endpoints

---

## Executive Summary

This report analyzes how three LLM providers (OpenRouter, OpenAI, and Corporate endpoints) track and report usage costs. Key findings:

1. **OpenRouter** provides the most comprehensive cost tracking with actual dollar amounts in API responses
2. **OpenAI** provides token counts but requires manual cost calculation based on pricing tables
3. **Corporate endpoint** currently has no cost tracking implementation (extends ChatOpenAI base)

---

## 1. OpenRouter API Cost Tracking

### 1.1 Overview

OpenRouter provides **real-time cost tracking** directly in API responses without requiring additional API calls. This is the most sophisticated cost tracking among the three providers.

### 1.2 Enabling Usage Tracking

Add the `usage` parameter to your request:

```json
{
  "model": "anthropic/claude-3.5-sonnet",
  "messages": [...],
  "usage": {
    "include": true
  }
}
```

### 1.3 Response Structure

```json
{
  "id": "gen-xxx",
  "choices": [...],
  "usage": {
    "prompt_tokens": 194,
    "completion_tokens": 2,
    "total_tokens": 196,
    "cached_tokens": 0,
    "cost": 0.95,
    "cost_details": {
      "upstream_inference_cost": 19
    }
  }
}
```

### 1.4 Key Fields

| Field | Type | Description |
|-------|------|-------------|
| `prompt_tokens` | int | Input tokens using model's native tokenizer |
| `completion_tokens` | int | Output tokens generated |
| `total_tokens` | int | Sum of prompt + completion tokens |
| `cached_tokens` | int | Number of tokens read from cache |
| `cost` | float | **Total charge to your account in USD** |
| `cost_details.upstream_inference_cost` | float | Actual cost from AI provider (in cents) |

### 1.5 Alternative: Generation API

For post-hoc cost retrieval:

```bash
GET https://openrouter.ai/api/v1/generation/{generation_id}
```

Returns full usage statistics including native tokenization and precise cost data.

### 1.6 Performance Impact

- Adds **200-300ms** to final response for token/cost calculation
- Only affects last message in streaming responses
- Uses model's native tokenizer for accuracy

### 1.7 Current Implementation Status

**File:** `/home/nick/python/autolabeler/src/autolabeler/core/llm_providers/openrouter.py`

**Current State:**
- Uses credit-based rate limiting
- Checks available credits via `/api/v1/auth/key` endpoint
- **Does NOT currently parse usage/cost from responses**
- No budget tracking implementation

**Credit Check Response:**
```json
{
  "data": {
    "credit_balance": 10.50,
    "usage": 5.25,
    "limit": 15.75
  }
}
```

**Observed Code:**
```python
# Lines 86-95: Current credit checking
usage = data.get("usage")
limit = data.get("limit")
if usage is not None and limit is not None:
    return float(limit) - float(usage)
```

**Missing:** No code to capture `usage` object from completion responses.

---

## 2. OpenAI API Cost Tracking

### 2.1 Overview

OpenAI provides **token counts** in every API response but **does not include cost amounts**. Developers must calculate costs manually using published pricing tables.

### 2.2 Response Structure (Chat Completions API)

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1704067200,
  "model": "gpt-4o",
  "choices": [...],
  "usage": {
    "prompt_tokens": 13,
    "completion_tokens": 7,
    "total_tokens": 20
  }
}
```

### 2.3 Response Structure (Responses API - 2025)

Newer API with expanded usage tracking:

```json
{
  "id": "resp-xxx",
  "object": "chat.completion",
  "usage": {
    "input_tokens": 405,
    "output_tokens": 285,
    "output_tokens_details": {
      "reasoning_tokens": 0
    },
    "total_tokens": 690
  }
}
```

### 2.4 Key Fields

| Field | Type | Description |
|-------|------|-------------|
| `prompt_tokens` / `input_tokens` | int | Tokens in the input prompt |
| `completion_tokens` / `output_tokens` | int | Tokens in model response |
| `total_tokens` | int | Sum of input + output |
| `output_tokens_details.reasoning_tokens` | int | Chain-of-thought tokens (o1 models) |

**Note:** Field names changed from `prompt_tokens`/`completion_tokens` (legacy) to `input_tokens`/`output_tokens` (2025 Responses API).

### 2.5 Cost Calculation

Costs must be calculated manually:

```python
def calculate_openai_cost(usage, model_pricing):
    """
    Example pricing (2025):
    - GPT-4o: $2.50 per 1M input tokens, $10.00 per 1M output tokens
    - GPT-4: $30.00 per 1M input tokens, $60.00 per 1M output tokens
    """
    input_cost = (usage["input_tokens"] / 1_000_000) * model_pricing["input_per_million"]
    output_cost = (usage["output_tokens"] / 1_000_000) * model_pricing["output_per_million"]
    return input_cost + output_cost
```

### 2.6 Usage Monitoring APIs

OpenAI provides separate monitoring APIs:

**Usage API:**
```bash
GET https://api.openai.com/v1/usage
```

Returns aggregated usage data with filtering by:
- Date range
- Model
- Project
- User ID

**Response includes:**
- `input_tokens`, `output_tokens`
- `num_model_requests`
- `input_cached_tokens`, `input_audio_tokens`, `output_audio_tokens`

**Cost API:**
```bash
GET https://api.openai.com/v1/costs
```

Returns actual dollar amounts for billing periods.

### 2.7 Current Implementation Status

**File:** `/home/nick/python/autolabeler/src/autolabeler/core/llm_providers/corporate.py`

**Current State:**
- Corporate endpoint extends `ChatOpenAI` from LangChain
- **No custom cost tracking implemented**
- Relies on base LangChain behavior
- LangChain ChatOpenAI **does** expose usage in `LLMResult.llm_output["token_usage"]`

**LangChain Usage Access:**
```python
# LangChain automatically parses OpenAI's usage field
result = client.generate(messages)
usage = result.llm_output.get("token_usage")
# Returns: {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20}
```

**Missing:** No budget tracking or cost calculation layer.

---

## 3. Corporate Endpoint Implementation

### 3.1 Overview

The Corporate endpoint is designed for internal/private LLM deployments with OpenAI-compatible APIs.

### 3.2 Current Implementation

**File:** `/home/nick/python/autolabeler/src/autolabeler/core/llm_providers/corporate.py`

**Features:**
- Extends `ChatOpenAI` from LangChain
- Security validations for internal URLs
- Custom headers for corporate authentication
- SSL verification options
- Structured output support

**Configuration:**
```python
client = CorporateOpenAIClient(
    api_key=os.getenv("CORPORATE_API_KEY"),
    base_url=os.getenv("CORPORATE_BASE_URL"),  # e.g., "https://llm.internal.company.com/v1"
    model="gpt-3.5-turbo"
)
```

### 3.3 Cost Tracking Status

**Current State:** ❌ **No cost tracking**

**Assumptions:**
- Corporate endpoints **may or may not** return OpenAI-compatible `usage` fields
- Some corporate deployments track costs internally
- Some may not expose cost data at all

**Challenges:**
1. **Variability:** Different corporate deployments may have different APIs
2. **Privacy:** Some corporations may not expose usage/cost for internal models
3. **Billing:** Internal deployments may use different billing models (fixed cost, no cost, etc.)

### 3.4 Recommendations for Corporate Endpoint

Given the variability, implement **optional cost tracking**:

```python
def get_usage_if_available(response):
    """Try to extract usage data, return None if not available."""
    try:
        if hasattr(response, "llm_output"):
            return response.llm_output.get("token_usage")
        return None
    except Exception:
        return None
```

---

## 4. Comparison Matrix

| Feature | OpenRouter | OpenAI | Corporate |
|---------|------------|--------|-----------|
| **Token Counts** | ✅ Native tokenizer | ✅ Standard | ⚠️ Varies |
| **Cost in Response** | ✅ Yes (USD) | ❌ No | ⚠️ Varies |
| **Cached Tokens** | ✅ Yes | ✅ Yes (2025) | ⚠️ Varies |
| **Reasoning Tokens** | ⚠️ Model-dependent | ✅ Yes (o1 models) | ⚠️ Varies |
| **Separate Usage API** | ✅ Yes (generation API) | ✅ Yes (usage/cost APIs) | ❌ No |
| **Real-time Cost** | ✅ Yes | ❌ Manual calc | ⚠️ Unknown |
| **Performance Impact** | ⚠️ +200-300ms | ✅ None | ✅ None |

---

## 5. Recommendations for Unified Budget Tracking

### 5.1 Design Goals

1. **Provider-agnostic interface:** Same API for all providers
2. **Graceful degradation:** Handle missing cost data
3. **Extensible:** Easy to add new providers
4. **Real-time tracking:** Track costs during processing
5. **Configurable budgets:** Set limits per provider/task

### 5.2 Proposed Architecture

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class UsageData:
    """Normalized usage data across all providers."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    provider: str = ""
    model: str = ""

class CostTracker(ABC):
    """Abstract base class for provider-specific cost tracking."""

    @abstractmethod
    def extract_usage(self, response) -> Optional[UsageData]:
        """Extract usage data from API response."""
        pass

    @abstractmethod
    def calculate_cost(self, usage: UsageData) -> float:
        """Calculate cost in USD from usage data."""
        pass

class OpenRouterCostTracker(CostTracker):
    """Cost tracker for OpenRouter API."""

    def extract_usage(self, response) -> Optional[UsageData]:
        """OpenRouter returns cost directly."""
        if not hasattr(response, "llm_output"):
            return None

        usage = response.llm_output.get("usage", {})
        return UsageData(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            cached_tokens=usage.get("cached_tokens"),
            cost_usd=usage.get("cost"),  # Already in USD!
            provider="openrouter",
            model=response.llm_output.get("model", "")
        )

    def calculate_cost(self, usage: UsageData) -> float:
        """Cost already provided by OpenRouter."""
        return usage.cost_usd or 0.0

class OpenAICostTracker(CostTracker):
    """Cost tracker for OpenAI API."""

    def __init__(self, pricing_table: dict):
        """Initialize with model pricing table."""
        self.pricing = pricing_table

    def extract_usage(self, response) -> Optional[UsageData]:
        """Extract token counts from OpenAI response."""
        if not hasattr(response, "llm_output"):
            return None

        usage = response.llm_output.get("token_usage", {})
        model = response.llm_output.get("model", "")

        return UsageData(
            input_tokens=usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0) or usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            reasoning_tokens=usage.get("output_tokens_details", {}).get("reasoning_tokens"),
            cost_usd=None,  # Must calculate
            provider="openai",
            model=model
        )

    def calculate_cost(self, usage: UsageData) -> float:
        """Calculate cost from token counts and pricing table."""
        model_pricing = self.pricing.get(usage.model, self.pricing.get("default"))
        input_cost = (usage.input_tokens / 1_000_000) * model_pricing["input_per_million"]
        output_cost = (usage.output_tokens / 1_000_000) * model_pricing["output_per_million"]
        return input_cost + output_cost

class CorporateCostTracker(CostTracker):
    """Cost tracker for corporate endpoints (best-effort)."""

    def extract_usage(self, response) -> Optional[UsageData]:
        """Try to extract usage if available."""
        try:
            if not hasattr(response, "llm_output"):
                return None

            usage = response.llm_output.get("token_usage", {})
            if not usage:
                return None

            return UsageData(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                cost_usd=usage.get("cost"),  # May or may not exist
                provider="corporate",
                model="corporate"
            )
        except Exception:
            return None

    def calculate_cost(self, usage: UsageData) -> float:
        """Return 0.0 for corporate (internal billing)."""
        return usage.cost_usd or 0.0

class BudgetManager:
    """Unified budget manager for all providers."""

    def __init__(self, budget_limit: float = float('inf')):
        self.budget_limit = budget_limit
        self.spent = 0.0
        self.usage_history = []
        self.trackers = {
            "openrouter": OpenRouterCostTracker(),
            "openai": OpenAICostTracker(self._get_openai_pricing()),
            "corporate": CorporateCostTracker()
        }

    def _get_openai_pricing(self) -> dict:
        """Load OpenAI pricing table (could be from config file)."""
        return {
            "gpt-4o": {"input_per_million": 2.50, "output_per_million": 10.00},
            "gpt-4": {"input_per_million": 30.00, "output_per_million": 60.00},
            "gpt-3.5-turbo": {"input_per_million": 0.50, "output_per_million": 1.50},
            "default": {"input_per_million": 1.00, "output_per_million": 2.00}
        }

    def track_request(self, response, provider: str) -> Optional[UsageData]:
        """Track a single API request."""
        tracker = self.trackers.get(provider)
        if not tracker:
            return None

        usage = tracker.extract_usage(response)
        if not usage:
            return None

        cost = tracker.calculate_cost(usage)
        usage.cost_usd = cost

        self.spent += cost
        self.usage_history.append(usage)

        return usage

    def check_budget(self) -> tuple[bool, float]:
        """Check if budget limit exceeded."""
        remaining = self.budget_limit - self.spent
        return remaining > 0, remaining

    def get_summary(self) -> dict:
        """Get spending summary by provider."""
        summary = {"total": self.spent, "by_provider": {}}
        for usage in self.usage_history:
            provider = usage.provider
            if provider not in summary["by_provider"]:
                summary["by_provider"][provider] = {"cost": 0.0, "requests": 0}
            summary["by_provider"][provider]["cost"] += usage.cost_usd or 0.0
            summary["by_provider"][provider]["requests"] += 1
        return summary
```

### 5.3 Usage Example

```python
# Initialize budget manager
budget = BudgetManager(budget_limit=10.00)  # $10 limit

# After each API call
response = client.invoke(messages)
usage = budget.track_request(response, provider="openrouter")

if usage:
    print(f"Request cost: ${usage.cost_usd:.4f}")
    print(f"Tokens: {usage.total_tokens} ({usage.input_tokens} in, {usage.output_tokens} out)")

# Check budget
within_budget, remaining = budget.check_budget()
if not within_budget:
    raise BudgetExceededError(f"Budget exceeded! Spent: ${budget.spent:.2f}")

# Get summary
summary = budget.get_summary()
print(f"Total spent: ${summary['total']:.2f}")
print(f"OpenRouter: ${summary['by_provider']['openrouter']['cost']:.2f}")
```

### 5.4 Integration with Existing Code

**Step 1:** Add usage tracking to OpenRouter client

```python
# In openrouter.py
class OpenRouterClient(ChatOpenAI):
    def _generate(self, *args, **kwargs):
        result = super()._generate(*args, **kwargs)

        # Store usage in llm_output if available
        if hasattr(result, "llm_output") and not result.llm_output.get("usage"):
            # Parse usage from response if present
            # OpenRouter returns usage in the response
            pass

        return result
```

**Step 2:** Add cost tracking to factory

```python
# In factory.py
def get_llm_client(settings: Settings, config: "LabelingConfig", budget_manager: Optional[BudgetManager] = None):
    client = _create_client(settings, config)

    if budget_manager:
        # Wrap client with budget tracking
        return BudgetTrackedClient(client, budget_manager, provider=_get_provider_name(settings))

    return client
```

---

## 6. Key Findings Summary

1. **OpenRouter** has the most complete cost tracking:
   - Direct USD cost in responses
   - Native tokenization
   - Cached token tracking
   - Minimal code changes needed

2. **OpenAI** requires manual cost calculation:
   - Token counts provided
   - Need pricing table
   - Supports reasoning tokens (o1 models)
   - Requires cost calculation layer

3. **Corporate** has unknown/variable support:
   - May or may not provide usage
   - Internal billing models vary
   - Best-effort tracking recommended
   - Fallback to token counting

4. **Unified interface is feasible:**
   - Abstract tracker pattern works across all providers
   - Graceful degradation for missing data
   - Real-time budget enforcement possible
   - Extensible for future providers

---

## 7. Action Items for Implementation

1. ✅ **Research complete** - All three providers documented
2. ⏭️ **Implement UsageData dataclass** - Normalized usage representation
3. ⏭️ **Implement CostTracker classes** - Provider-specific extractors
4. ⏭️ **Implement BudgetManager** - Unified tracking interface
5. ⏭️ **Update OpenRouter client** - Parse usage from responses
6. ⏭️ **Update factory** - Integrate budget tracking
7. ⏭️ **Add OpenAI pricing config** - Pricing table for cost calculation
8. ⏭️ **Add tests** - Verify cost tracking accuracy
9. ⏭️ **Add logging** - Track budget usage in logs
10. ⏭️ **Add CLI flags** - Budget limits via command line

---

## 8. References

### OpenRouter Documentation
- Usage Accounting: https://openrouter.ai/docs/use-cases/usage-accounting
- API Reference: https://openrouter.ai/docs/api-reference/overview
- Generation API: https://openrouter.ai/docs/api-reference/generation

### OpenAI Documentation
- Chat Completions API: https://platform.openai.com/docs/api-reference/chat
- Responses API (2025): https://platform.openai.com/docs/guides/responses
- Usage API: https://cookbook.openai.com/examples/completions_usage_api
- Pricing: https://openai.com/api/pricing/

### Codebase Files
- `/home/nick/python/autolabeler/src/autolabeler/core/llm_providers/openrouter.py`
- `/home/nick/python/autolabeler/src/autolabeler/core/llm_providers/corporate.py`
- `/home/nick/python/autolabeler/src/autolabeler/core/llm_providers/factory.py`

---

**Research Status:** ✅ Complete
**Next Phase:** Implementation (Coder Agent)
**Coordination Key:** `hive/research/cost-tracking`
