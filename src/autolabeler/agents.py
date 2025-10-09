"""Multi-agent service for coordinating simultaneous classification tasks."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from .config import Settings


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
        self._initialize_client()

    def _initialize_client(self):
        """Initialize LLM client based on settings."""
        if self.settings.llm_provider == "openrouter":
            from .openrouter_client import OpenRouterClient

            self.client = OpenRouterClient(
                api_key=self.settings.openrouter_api_key,
                model=self.settings.llm_model,
                temperature=self.settings.temperature,
                use_rate_limiter=True,  # Enable 500 req/sec rate limiting
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
        else:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.settings.openai_api_key)
            self.is_anthropic = False
            self.is_openrouter = False

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
                    # OpenRouter uses OpenAI-compatible interface
                    response = self.client.create(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=2048,
                        response_format=json_schema,
                    )
                    return response["choices"][0]["message"]["content"]
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

    def label_with_agents(
        self, df: pd.DataFrame, text_column: str, tasks: list[str]
    ) -> pd.DataFrame:
        """Label entire DataFrame with multiple classification tasks.

        Args:
            df: DataFrame with text to classify
            text_column: Name of column containing text
            tasks: List of task names to perform

        Returns:
            DataFrame with added columns for each task's label and confidence
        """
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
