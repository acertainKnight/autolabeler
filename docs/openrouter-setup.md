# OpenRouter Integration Setup

## Overview

The autolabeler now supports OpenRouter for LLM routing with automatic rate limiting up to 500 requests/second based on available credits.

## Configuration

### 1. Set Environment Variables

Create or update your `.env` file:

```bash
# OpenRouter Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here
LLM_PROVIDER=openrouter
LLM_MODEL=anthropic/claude-3.5-sonnet
TEMPERATURE=0.1
```

### 2. Supported Models

OpenRouter supports multiple providers:
- **Anthropic**: `anthropic/claude-3.5-sonnet`, `anthropic/claude-3-opus`
- **OpenAI**: `openai/gpt-4`, `openai/gpt-3.5-turbo`
- **Meta**: `meta-llama/llama-3.1-8b-instruct:free` (free tier)
- **And many more**: See https://openrouter.ai/models

### 3. Rate Limiting

The OpenRouter client automatically:
1. Checks your available credits via the OpenRouter API
2. Configures rate limiting: `min(available_credits, 500)` requests/second
3. Blocks requests when rate limit is exceeded (automatic throttling)

**Credit Formula:**
- If credits ≥ 1: Rate limit = min(credits, 500) req/sec
- If credits < 1: Rate limit = 1 req/sec (minimum)
- If unlimited: Rate limit = 500 req/sec (maximum)

### 4. Usage in Fed Headlines Pipeline

The Fed headlines pipeline automatically uses OpenRouter when configured:

```bash
# Ensure environment is set
export OPENROUTER_API_KEY=your_key
export LLM_PROVIDER=openrouter
export LLM_MODEL=anthropic/claude-3.5-sonnet

# Run the pipeline (uses OpenRouter automatically)
./scripts/fed_headlines_manual_run.sh complete
```

## Implementation Details

### Files Modified

1. **src/autolabeler/openrouter_client.py** (NEW)
   - `OpenRouterRateLimiter`: Credit-based rate limiting
   - `OpenRouterClient`: OpenAI-compatible interface
   - Adapted from project-thoth implementation

2. **src/autolabeler/agents.py**
   - Updated `_initialize_client()` to support OpenRouter provider
   - Updated `_call_llm()` to handle OpenRouter responses
   - Added `is_openrouter` flag for routing

3. **src/autolabeler/active_learning.py**
   - Updated `_generate_rule_from_llm()` to support OpenRouter
   - Rate limiting applies to rule generation calls

4. **src/autolabeler/config.py**
   - Added `llm_provider: str = "openrouter"` (default)
   - Added `temperature: float = 0.1`
   - Already had `openrouter_api_key` and `openrouter_base_url`

### API Endpoints Used

- **Chat Completions**: `POST https://openrouter.ai/api/v1/chat/completions`
- **Credit Check**: `GET https://openrouter.ai/api/v1/auth/key`
- **Models List**: `GET https://openrouter.ai/api/v1/models`

### Example Usage in Code

```python
from autolabeler.config import Settings
from autolabeler.openrouter_client import OpenRouterClient

# Initialize with rate limiting
settings = Settings()  # Loads from .env
client = OpenRouterClient(
    api_key=settings.openrouter_api_key,
    model=settings.llm_model,
    temperature=settings.temperature,
    use_rate_limiter=True  # Enables 500 req/sec limiting
)

# Make a request (automatically rate limited)
response = client.create(
    messages=[{"role": "user", "content": "Classify this text..."}]
)
print(response["choices"][0]["message"]["content"])
```

## Performance Characteristics

### High-Throughput Labeling

With 500 req/sec capability:
- **20,000 labels**: ~40 seconds (at max rate)
- **100,000 labels**: ~3.3 minutes (at max rate)

**Actual timing** will be slower due to:
- Model inference time
- Batch processing overhead
- Rate limiting based on available credits

### Cost Optimization

OpenRouter provides:
- **Free models**: `meta-llama/llama-3.1-8b-instruct:free`
- **Competitive pricing**: Often cheaper than direct API access
- **Transparent costs**: Per-token pricing visible in dashboard

## Troubleshooting

### "Unable to determine credits"

**Cause**: API key check failed or invalid key

**Fix**:
```bash
# Verify your API key
curl https://openrouter.ai/api/v1/auth/key \
  -H "Authorization: Bearer $OPENROUTER_API_KEY"

# Should return: {"data": {"limit": X, "usage": Y, ...}}
```

### Rate Limiting Too Slow

**Cause**: Insufficient credits

**Fix**:
1. Check credits at https://openrouter.ai/credits
2. Add credits to your account
3. Rate limiter will auto-adjust on next request

### "Module not found" Errors

**Cause**: Dependencies not installed

**Fix**:
```bash
pip install requests loguru pydantic-settings
# Or: uv pip install -r requirements.txt
```

## Migration from Anthropic/OpenAI

### Before (Direct Anthropic)
```python
from anthropic import Anthropic
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
```

### After (OpenRouter)
```bash
# .env file
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_key
LLM_MODEL=anthropic/claude-3.5-sonnet  # Same model via OpenRouter
```

**No code changes needed** - the autolabeler automatically detects `llm_provider` setting.

## References

- **OpenRouter Docs**: https://openrouter.ai/docs
- **Model List**: https://openrouter.ai/models
- **Pricing**: https://openrouter.ai/docs/pricing
- **Credits Dashboard**: https://openrouter.ai/credits
