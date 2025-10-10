"""Multi-agent service for coordinating simultaneous classification tasks."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from .config import Settings

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not installed. Async processing will not be available. Install with: pip install httpx")


class TaskConfig(BaseModel):
    """Configuration for a single classification task."""

    type: str = "classification"
    labels: list[str]
    principles: list[str] = Field(default_factory=list)
    description: str | None = None


class MultiLabelResult(BaseModel):
    """Result from multi-label classification."""

    labels: dict[str, str]
    confidences: dict[str, float]
    reasoning: dict[str, str] | None = None


class MultiAgentService:
    """Coordinate multiple classification tasks simultaneously."""

    def __init__(self, settings: Settings, task_configs: dict[str, dict[str, Any]]):
        """Initialize multi-agent service.

        Args:
            settings: AutoLabeler settings for LLM configuration
            task_configs: Dictionary mapping task names to their configurations
                         Each config should have: type, labels, principles
        """
        self.settings = settings
        self.task_configs = {
            name: TaskConfig(**config) for name, config in task_configs.items()
        }
        self.cost_tracker = None  # Will be initialized if budget is set
        self._initialize_client()

    def _initialize_client(self):
        """Initialize LLM client based on settings with cost tracking."""
        from .core.utils.budget_tracker import CostTracker

        # Create cost tracker if budget is specified
        if self.settings.llm_budget is not None:
            self.cost_tracker = CostTracker(budget=self.settings.llm_budget)
            logger.info(f"Cost tracking enabled with budget: ${self.settings.llm_budget:.2f}")

        if self.settings.llm_provider == "openrouter":
            from .core.llm_providers.openrouter import OpenRouterClient

            self.client = OpenRouterClient(
                api_key=self.settings.openrouter_api_key,
                model=self.settings.llm_model,
                temperature=self.settings.temperature,
                use_rate_limiter=True,  # Enable 500 req/sec rate limiting
                cost_tracker=self.cost_tracker,  # Pass cost tracker to client
            )
            self.is_anthropic = False
            self.is_openrouter = True
            logger.info(
                f"Initialized OpenRouter client with model {self.settings.llm_model} and rate limiting"
            )
        elif self.settings.llm_provider == "anthropic":
            from anthropic import Anthropic

            self.client = Anthropic(api_key=self.settings.anthropic_api_key)
            self.is_anthropic = True
            self.is_openrouter = False
            if self.cost_tracker:
                logger.warning("Cost tracking not yet implemented for Anthropic provider")
        else:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.settings.openai_api_key)
            self.is_anthropic = False
            self.is_openrouter = False
            if self.cost_tracker:
                logger.warning("Cost tracking not yet implemented for OpenAI direct provider")

    def _build_prompt(self, text: str, tasks: list[str]) -> str:
        """Build structured prompt for multi-label classification.

        Args:
            text: Text to classify
            tasks: List of task names to perform

        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            "You are a multi-task classification system. Analyze the following text and provide classifications for each requested task.",
            f"\nText to analyze:\n{text}\n",
            "\nTasks to perform:",
        ]

        for task_name in tasks:
            if task_name not in self.task_configs:
                logger.warning(f"Task '{task_name}' not found in configs, skipping")
                continue

            config = self.task_configs[task_name]
            prompt_parts.append(f"\n{task_name.upper()}:")
            prompt_parts.append(f"  Valid labels: {', '.join(config.labels)}")

            if config.principles:
                prompt_parts.append("  Guidelines:")
                for principle in config.principles:
                    prompt_parts.append(f"    - {principle}")

        prompt_parts.append(
            "\nProvide your response as a JSON object with the following structure:"
        )
        prompt_parts.append("{")
        for task_name in tasks:
            if task_name in self.task_configs:
                prompt_parts.append(f'  "{task_name}": {{')
                prompt_parts.append('    "label": "chosen_label",')
                prompt_parts.append('    "confidence": 0.0-1.0,')
                prompt_parts.append('    "reasoning": "brief explanation"')
                prompt_parts.append("  },")
        prompt_parts.append("}")

        return "\n".join(prompt_parts)

    def _build_json_schema(self, tasks: list[str]) -> dict[str, Any]:
        """Build JSON schema for structured output.

        Args:
            tasks: List of task names to include in schema

        Returns:
            JSON schema dictionary for response_format parameter
        """
        properties = {}
        required_fields = []

        for task_name in tasks:
            if task_name not in self.task_configs:
                continue

            properties[task_name] = {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": f"Classification label for {task_name}",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Confidence score between 0 and 1",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation for the classification",
                    },
                },
                "required": ["label", "confidence", "reasoning"],
                "additionalProperties": False,
            }
            required_fields.append(task_name)

        return {
            "type": "json_schema",
            "json_schema": {
                "name": "multi_label_classification",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required_fields,
                    "additionalProperties": False,
                },
            },
        }

    def _parse_response(self, response_text: str, tasks: list[str]) -> MultiLabelResult:
        """Parse LLM response into structured result.

        Args:
            response_text: Raw LLM response
            tasks: List of task names

        Returns:
            MultiLabelResult with labels, confidences, and reasoning
        """
        try:
            # Extract JSON from response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")

            json_str = response_text[json_start:json_end]
            parsed = json.loads(json_str)

            labels = {}
            confidences = {}
            reasoning = {}

            for task_name in tasks:
                if task_name in parsed:
                    task_result = parsed[task_name]
                    # Convert label to string (LLM may return int for numeric labels)
                    label = task_result.get("label", "unknown")
                    labels[task_name] = str(label) if label is not None else "unknown"
                    confidences[task_name] = float(
                        task_result.get("confidence", 0.0)
                    )
                    reasoning[task_name] = task_result.get("reasoning", "")
                else:
                    logger.warning(f"Task '{task_name}' not found in response")
                    labels[task_name] = "unknown"
                    confidences[task_name] = 0.0
                    reasoning[task_name] = "No response from LLM"

            return MultiLabelResult(
                labels=labels, confidences=confidences, reasoning=reasoning
            )

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Response text: {response_text}")
            return MultiLabelResult(
                labels={task: "error" for task in tasks},
                confidences={task: 0.0 for task in tasks},
                reasoning={task: f"Parse error: {str(e)}" for task in tasks},
            )

    async def _call_llm_async(self, prompt: str, tasks: list[str]) -> str:
        """Make async LLM call for all tasks with structured output support.

        Args:
            prompt: Structured prompt with all tasks
            tasks: List of task names for schema generation

        Returns:
            LLM response text (JSON string)
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for async processing. Install with: pip install httpx")

        try:
            if self.is_openrouter:
                # Check budget before making call
                if self.cost_tracker and self.cost_tracker.is_budget_exceeded():
                    from .core.utils.budget_tracker import BudgetExceededError
                    stats = self.cost_tracker.get_stats()
                    raise BudgetExceededError(stats["total_cost"], stats["budget"])

                # OpenRouter async call
                json_schema = self._build_json_schema(tasks)

                async with httpx.AsyncClient(timeout=60.0) as client:
                    # Rate limiting
                    if hasattr(self.client, 'rate_limiter'):
                        self.client.rate_limiter.acquire()

                    response = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.settings.llm_model,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": self.settings.temperature,
                            "max_tokens": 2048,
                            "response_format": json_schema,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()

                    # Track cost after successful call
                    if self.cost_tracker:
                        from .core.utils.budget_tracker import extract_cost_from_result, BudgetExceededError
                        from langchain_core.outputs import ChatGeneration, LLMResult
                        from langchain_core.messages import AIMessage

                        # Convert httpx response to LLMResult format for cost extraction
                        message = AIMessage(
                            content=data["choices"][0]["message"]["content"],
                            response_metadata=data.get("usage", {})
                        )

                        # Add total_cost if available in response
                        if "usage" in data and "total_cost" in data["usage"]:
                            message.response_metadata["total_cost"] = data["usage"]["total_cost"]

                        generation = ChatGeneration(message=message)
                        llm_result = LLMResult(generations=[[generation]])

                        cost = extract_cost_from_result(llm_result, "openrouter", self.settings.llm_model)
                        if cost > 0:
                            within_budget = self.cost_tracker.add_cost(cost)
                            logger.debug(f"Tracked async call cost: ${cost:.6f}")

                            # Check if budget exceeded after adding cost
                            if not within_budget:
                                stats = self.cost_tracker.get_stats()
                                raise BudgetExceededError(stats["total_cost"], stats["budget"])

                    return data["choices"][0]["message"]["content"]
            else:
                # Fallback to sync for non-OpenRouter
                return self._call_llm(prompt, tasks)

        except Exception as e:
            logger.error(f"Async LLM call failed: {e}")
            return "{}"

    def _call_llm(self, prompt: str, tasks: list[str]) -> str:
        """Make single LLM call for all tasks with structured output support.

        Args:
            prompt: Structured prompt with all tasks
            tasks: List of task names for schema generation

        Returns:
            LLM response text (JSON string)
        """
        try:
            if self.is_openrouter or not self.is_anthropic:
                # OpenRouter and OpenAI support structured outputs
                json_schema = self._build_json_schema(tasks)

                if self.is_openrouter:
                    # OpenRouter uses LangChain's invoke interface
                    from langchain_core.messages import HumanMessage
                    messages = [HumanMessage(content=prompt)]
                    response = self.client.invoke(messages)
                    return response.content
                else:
                    # Direct OpenAI client
                    response = self.client.chat.completions.create(
                        model=self.settings.llm_model,
                        temperature=self.settings.temperature,
                        messages=[{"role": "user", "content": prompt}],
                        response_format=json_schema,
                    )
                    return response.choices[0].message.content
            else:
                # Anthropic doesn't support structured outputs, use text parsing
                response = self.client.messages.create(
                    model=self.settings.llm_model,
                    max_tokens=2048,
                    temperature=self.settings.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "{}"

    def label_single(self, text: str, tasks: list[str]) -> MultiLabelResult:
        """Label a single text with multiple classification tasks.

        Args:
            text: Text to classify
            tasks: List of task names to perform

        Returns:
            MultiLabelResult with all classifications
        """
        prompt = self._build_prompt(text, tasks)
        response = self._call_llm(prompt, tasks)
        return self._parse_response(response, tasks)

    async def _label_single_async(self, text: str, tasks: list[str]) -> MultiLabelResult:
        """Async version of label_single for concurrent processing.

        Args:
            text: Text to classify
            tasks: List of task names to perform

        Returns:
            MultiLabelResult with all classifications
        """
        prompt = self._build_prompt(text, tasks)
        response = await self._call_llm_async(prompt, tasks)
        return self._parse_response(response, tasks)

    def label_with_agents(
        self, df: pd.DataFrame, text_column: str, tasks: list[str], use_async: bool = True
    ) -> pd.DataFrame:
        """Label entire DataFrame with multiple classification tasks.

        Args:
            df: DataFrame with text to classify
            text_column: Name of column containing text
            tasks: List of task names to perform
            use_async: Use async concurrent processing (default: True)

        Returns:
            DataFrame with added columns for each task's label and confidence
        """
        if use_async and HTTPX_AVAILABLE and self.is_openrouter:
            return asyncio.run(self._label_with_agents_async(df, text_column, tasks))
        else:
            if use_async and not HTTPX_AVAILABLE:
                logger.warning("httpx not available, falling back to sequential processing")
            return self._label_with_agents_sync(df, text_column, tasks)

    async def _label_with_agents_async(
        self, df: pd.DataFrame, text_column: str, tasks: list[str]
    ) -> pd.DataFrame:
        """Async concurrent labeling of entire DataFrame.

        Processes all rows concurrently for maximum throughput.
        """
        logger.info(
            f"Starting async multi-label classification with tasks: {', '.join(tasks)}"
        )
        results_df = df.copy()

        # Initialize result columns
        for task_name in tasks:
            results_df[f"label_{task_name}"] = None
            results_df[f"confidence_{task_name}"] = 0.0
            results_df[f"reasoning_{task_name}"] = None

        # Create async tasks for all rows
        async_tasks = []
        indices = []
        for idx, row in df.iterrows():
            text = row[text_column]
            async_tasks.append(self._label_single_async(text, tasks))
            indices.append(idx)

        # Process all concurrently
        logger.info(f"Processing {len(async_tasks)} rows concurrently...")
        try:
            results = await asyncio.gather(*async_tasks, return_exceptions=False)
        except Exception as e:
            # Check if it's a budget error - if so, re-raise to stop processing
            from .core.utils.budget_tracker import BudgetExceededError
            if isinstance(e, BudgetExceededError):
                logger.warning(f"Budget exceeded during batch processing: {e}")
                raise
            # For other errors, log and continue
            logger.error(f"Error during concurrent processing: {e}")
            raise

        # Store results
        for idx, result in zip(indices, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process row {idx}: {result}")
                continue

            for task_name in tasks:
                results_df.at[idx, f"label_{task_name}"] = result.labels[task_name]
                results_df.at[idx, f"confidence_{task_name}"] = result.confidences[task_name]
                if result.reasoning:
                    results_df.at[idx, f"reasoning_{task_name}"] = result.reasoning[task_name]

        logger.info(f"Async multi-label classification complete: {len(df)} rows processed")
        return results_df

    def _label_with_agents_sync(
        self, df: pd.DataFrame, text_column: str, tasks: list[str]
    ) -> pd.DataFrame:
        """Synchronous sequential labeling (fallback when async not available)."""
        logger.info(
            f"Starting multi-label classification with tasks: {', '.join(tasks)}"
        )
        results_df = df.copy()

        # Initialize result columns
        for task_name in tasks:
            results_df[f"label_{task_name}"] = None
            results_df[f"confidence_{task_name}"] = 0.0
            results_df[f"reasoning_{task_name}"] = None

        # Process each row
        for row_num, (idx, row) in enumerate(df.iterrows(), start=1):
            text = row[text_column]
            result = self.label_single(text, tasks)

            # Store results
            for task_name in tasks:
                results_df.at[idx, f"label_{task_name}"] = result.labels[task_name]
                results_df.at[idx, f"confidence_{task_name}"] = result.confidences[
                    task_name
                ]
                if result.reasoning:
                    results_df.at[idx, f"reasoning_{task_name}"] = result.reasoning[
                        task_name
                    ]

            if row_num % 10 == 0:
                logger.info(f"Processed {row_num}/{len(df)} rows")

        logger.info("Multi-label classification complete")
        return results_df

    def update_task_configs(self, updated_rules: dict[str, list[str]]):
        """Update task configurations with new rules.

        Args:
            updated_rules: Dictionary mapping task names to updated principle lists
        """
        for task_name, principles in updated_rules.items():
            if task_name in self.task_configs:
                self.task_configs[task_name].principles = principles
                logger.info(f"Updated principles for task '{task_name}'")
            else:
                logger.warning(f"Task '{task_name}' not found, cannot update")

    def get_cost_stats(self) -> dict[str, Any] | None:
        """Get cost tracking statistics.

        Returns:
            Dictionary with cost statistics or None if cost tracking not enabled
        """
        if self.cost_tracker:
            return self.cost_tracker.get_stats()
        return None

    def log_cost_summary(self):
        """Log a summary of costs incurred during labeling."""
        if self.cost_tracker:
            stats = self.cost_tracker.get_stats()
            logger.info("=" * 60)
            logger.info("COST TRACKING SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total API Calls: {stats['call_count']}")
            logger.info(f"Total Cost: ${stats['total_cost']:.4f}")
            if stats['budget'] is not None:
                logger.info(f"Budget Limit: ${stats['budget']:.2f}")
                logger.info(f"Remaining Budget: ${stats['remaining_budget']:.4f}")
                budget_used_pct = (stats['total_cost'] / stats['budget']) * 100
                logger.info(f"Budget Used: {budget_used_pct:.1f}%")
            logger.info("=" * 60)
        else:
            logger.info("Cost tracking not enabled (no budget set)")
