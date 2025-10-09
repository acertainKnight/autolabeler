# Cost Tracking API Research - Executive Summary

**Research Date:** 2025-10-08
**Status:** ✅ Complete
**Memory Key:** `hive/research/cost-tracking`

---

## Quick Reference: API Response Structures

### OpenRouter API Response

```json
{
  "id": "gen-xxx",
  "choices": [...],
  "usage": {
    "prompt_tokens": 194,
    "completion_tokens": 2,
    "total_tokens": 196,
    "cached_tokens": 0,
    "cost": 0.95,                    // ✅ Direct USD cost
    "cost_details": {
      "upstream_inference_cost": 19  // Cost in cents from provider
    }
  }
}
```

**Enable with:** Add `"usage": {"include": true}` to request

### OpenAI API Response (2025)

```json
{
  "id": "chatcmpl-xxx",
  "usage": {
    "input_tokens": 405,
    "output_tokens": 285,
    "output_tokens_details": {
      "reasoning_tokens": 0
    },
    "total_tokens": 690
    // ❌ No cost field - must calculate manually
  }
}
```

**Manual Calculation Required:**
```python
cost = (input_tokens / 1_000_000 * input_price_per_million) +
       (output_tokens / 1_000_000 * output_price_per_million)
```

### Corporate Endpoint Response

```json
{
  "usage": {
    "prompt_tokens": 13,
    "completion_tokens": 7,
    "total_tokens": 20
    // ⚠️ May or may not include cost (varies by deployment)
  }
}
```

**Status:** Best-effort extraction, assumes zero cost for internal models

---

## Implementation Checklist

- [x] Research OpenRouter cost tracking
- [x] Research OpenAI cost tracking
- [x] Analyze corporate endpoint implementation
- [x] Document API response structures
- [x] Design unified interface
- [ ] Implement `UsageData` dataclass
- [ ] Implement provider-specific extractors
- [ ] Implement `BudgetManager` class
- [ ] Update OpenRouter client to parse usage
- [ ] Add OpenAI pricing configuration
- [ ] Integration tests
- [ ] CLI flags for budget limits

---

## Key Recommendations

1. **Use OpenRouter for built-in cost tracking** - No calculation needed
2. **Implement unified `BudgetManager` interface** - Works across all providers
3. **Graceful degradation** - Handle missing cost data elegantly
4. **Real-time budget enforcement** - Check before each API call
5. **Provider-specific extractors** - Abstract the differences

---

## Code Location

- **Full Report:** `/home/nick/python/autolabeler/docs/research/cost-tracking-api-research.md`
- **OpenRouter Client:** `/home/nick/python/autolabeler/src/autolabeler/core/llm_providers/openrouter.py`
- **Corporate Client:** `/home/nick/python/autolabeler/src/autolabeler/core/llm_providers/corporate.py`
- **Factory:** `/home/nick/python/autolabeler/src/autolabeler/core/llm_providers/factory.py`

---

## Next Steps

**For Coder Agent:**
1. Implement the `BudgetManager` class from the architecture in the full report
2. Add usage extraction to OpenRouter client
3. Create OpenAI pricing configuration file
4. Integrate with factory pattern

**For Tester Agent:**
1. Create unit tests for cost extraction
2. Test budget enforcement
3. Mock API responses with various usage structures
4. Verify graceful degradation

---

**Research Complete** | Coordination hooks executed | Findings stored in collective memory
