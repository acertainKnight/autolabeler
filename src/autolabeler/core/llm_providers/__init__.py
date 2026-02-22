"""
LLM provider clients for AutoLabeler.

Unified provider interface for OpenAI, Anthropic, Google, and OpenRouter.

Custom providers can be integrated by subclassing ``BaseLLMProvider`` and
registering them with ``register_provider()``:

    from autolabeler.core.llm_providers import BaseLLMProvider, register_provider, LLMResponse

    class MyCorporateProxy(BaseLLMProvider):
        def __init__(self, model: str):
            self.model = model

        async def call(self, system, user, temperature, logprobs=False, response_schema=None):
            text = await my_internal_api(system, user)
            return LLMResponse(
                text=text,
                parsed_json=self.parse_json_response(text),
                logprobs=None,
                cost=0.0,
                model_name=self.model,
            )

    register_provider("corporate_proxy", MyCorporateProxy)
"""

from .providers import (
    BaseLLMProvider,
    LLMProvider,
    LLMResponse,
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    OpenRouterProvider,
    get_provider,
    get_registered_providers,
    load_provider_module,
    register_provider,
)

__all__ = [
    "BaseLLMProvider",
    "LLMProvider",
    "LLMResponse",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "OpenRouterProvider",
    "get_provider",
    "get_registered_providers",
    "load_provider_module",
    "register_provider",
]
