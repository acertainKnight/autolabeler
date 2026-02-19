"""Unified labeling pipeline for any dataset.

This is the core of the autolabeler service. One pipeline handles all datasets,
configured by DatasetConfig and PromptRegistry.

5-Stage Architecture:
1. Optional relevancy gate (cheap pre-filter)
2. Heterogeneous jury (parallel model calls)
3. Confidence-weighted aggregation
4. Optional candidate annotation (for disagreements)
5. Tier assignment (ACCEPT, ACCEPT-M, SOFT, QUARANTINE)
"""

import asyncio
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from autolabeler.core.dataset_config import DatasetConfig, ModelConfig
from autolabeler.core.llm_providers.providers import get_provider, LLMResponse
from autolabeler.core.prompts.registry import PromptRegistry
from autolabeler.core.quality.confidence_scorer import ConfidenceScorer


@dataclass
class LabelResult:
    """Result of labeling a single text.
    
    Attributes:
        label: Primary label (most likely)
        label_type: "hard" or "soft"
        tier: ACCEPT, ACCEPT-M, SOFT, or QUARANTINE
        training_weight: Weight for training (0-1)
        agreement: Agreement type (unanimous, majority_adjacent, etc.)
        jury_labels: List of labels from each jury member
        jury_confidences: List of confidences from each jury member
        soft_label: Optional soft label distribution {label: prob}
        reasoning: Reasoning from jury members
        error: Optional error message
    """
    label: int | str | None
    label_type: str  # "hard" or "soft"
    tier: str  # ACCEPT, ACCEPT-M, SOFT, QUARANTINE
    training_weight: float
    agreement: str
    jury_labels: list[Any]
    jury_confidences: list[float]
    soft_label: dict[Any, float] | None = None
    reasoning: list[Any] | None = None
    error: str | None = None


class LabelingPipeline:
    """Unified labeling pipeline for any dataset.
    
    Example:
        >>> config = DatasetConfig.from_yaml("configs/fed_headlines.yaml")
        >>> registry = PromptRegistry("fed_headlines")
        >>> pipeline = LabelingPipeline(config, registry)
        >>> 
        >>> # Label one text
        >>> result = await pipeline.label_one("FED SEES RATES RISING")
        >>> print(result.label, result.tier)
        >>> 
        >>> # Label dataframe
        >>> df = pd.read_csv("data.csv")
        >>> results_df = await pipeline.label_dataframe(df, "output.csv")
    """
    
    def __init__(
        self,
        dataset_config: DatasetConfig,
        prompt_registry: PromptRegistry,
        confidence_scorer: ConfidenceScorer | None = None,
    ):
        """Initialize pipeline.
        
        Parameters:
            dataset_config: Configuration for this dataset
            prompt_registry: Prompt registry for this dataset
            confidence_scorer: Optional pre-calibrated confidence scorer
        """
        self.config = dataset_config
        self.prompts = prompt_registry
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()
        
        # Initialize providers
        self.jury_providers = [
            get_provider(m.provider, m.model)
            for m in self.config.jury_models
        ]
        
        self.gate_provider = None
        if self.config.gate_model:
            self.gate_provider = get_provider(
                self.config.gate_model.provider,
                self.config.gate_model.model
            )
        
        self.candidate_provider = None
        if self.config.candidate_model:
            self.candidate_provider = get_provider(
                self.config.candidate_model.provider,
                self.config.candidate_model.model
            )
        
        logger.info(
            f"Initialized LabelingPipeline for {self.config.name} with "
            f"{len(self.jury_providers)} jury models"
        )
    
    async def label_one(self, text: str, rag_examples: str = "") -> LabelResult:
        """Label a single text through the 5-stage pipeline.
        
        Parameters:
            text: Text to label
            rag_examples: Optional RAG-retrieved examples
            
        Returns:
            LabelResult with label, confidence, tier, etc.
        """
        try:
            # Stage 1: Optional relevancy gate (skip for now - jury handles it)
            # Could add if config.use_relevancy_gate later
            
            # Stage 2: Jury voting
            system_prompt, user_prompt = self.prompts.build_labeling_prompt(
                text, rag_examples
            )
            
            jury_results = await self._call_jury(system_prompt, user_prompt)
            
            if len(jury_results) < 2:
                return LabelResult(
                    label=None,
                    label_type="hard",
                    tier="QUARANTINE",
                    training_weight=0.0,
                    agreement="jury_failure",
                    jury_labels=[],
                    jury_confidences=[],
                    error="Jury failed (< 2 models responded)"
                )
            
            # Stage 3: Confidence-weighted aggregation
            return await self._aggregate_votes(text, jury_results, rag_examples)
        
        except Exception as e:
            logger.error(f"Pipeline failed for text: {e}")
            return LabelResult(
                label=None,
                label_type="hard",
                tier="QUARANTINE",
                training_weight=0.0,
                agreement="error",
                jury_labels=[],
                jury_confidences=[],
                error=str(e)
            )
    
    async def _call_jury(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> list[dict[str, Any]]:
        """Call all jury models in parallel.
        
        Returns:
            List of results from jury members
        """
        tasks = []
        for i, provider in enumerate(self.jury_providers):
            model_config = self.config.jury_models[i]
            tasks.append(
                self._call_one_juror(
                    provider,
                    model_config,
                    system_prompt,
                    user_prompt
                )
            )
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        jury_results = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.warning(f"Juror {i} failed: {r}")
                continue
            r["_model_config"] = self.config.jury_models[i]
            jury_results.append(r)
        
        return jury_results
    
    async def _call_one_juror(
        self,
        provider: Any,
        model_config: ModelConfig,
        system_prompt: str,
        user_prompt: str
    ) -> dict[str, Any]:
        """Call one jury member.
        
        Returns:
            Dict with label, confidence, reasoning, logprobs
        """
        response: LLMResponse = await provider.call(
            system=system_prompt,
            user=user_prompt,
            temperature=self.config.jury_temperature,
            logprobs=model_config.has_logprobs,
        )
        
        result = {
            "label": None,
            "confidence": 0.5,
            "reasoning": None,
            "logprobs": response.logprobs,
            "model_name": model_config.name,
        }
        
        if response.parsed_json:
            result["label"] = response.parsed_json.get("label")
            result["reasoning"] = response.parsed_json.get("reasoning")
            verbal_conf = response.parsed_json.get("confidence", "medium")
            if isinstance(verbal_conf, str):
                result["confidence"] = self.confidence_scorer.from_verbal(verbal_conf)
            elif isinstance(verbal_conf, (int, float)):
                result["confidence"] = float(verbal_conf)
        
        return result
    
    async def _aggregate_votes(
        self,
        text: str,
        jury_results: list[dict],
        rag_examples: str
    ) -> LabelResult:
        """Stage 3: Aggregate jury votes with confidence weighting.
        
        Also handles stages 4-5: candidate annotation and tier assignment.
        """
        # Extract labels and calculate confidence-weighted votes
        votes = []
        for r in jury_results:
            label = r["label"]
            
            # Get confidence
            model_config = r["_model_config"]
            if model_config.has_logprobs and r.get("logprobs"):
                raw_conf = self.confidence_scorer.from_logprobs(
                    r["logprobs"], label
                )
            else:
                raw_conf = r.get("confidence", 0.7)
            
            # Calibrate if available
            conf = self.confidence_scorer.calibrate(raw_conf)
            
            votes.append({
                "label": label,
                "confidence": conf,
                "model": model_config.name
            })
        
        # Weighted majority vote
        label_weights: dict[Any, float] = {}
        for v in votes:
            label = v["label"]
            label_weights[label] = label_weights.get(label, 0) + v["confidence"]
        
        winning_label = max(label_weights, key=label_weights.get)
        labels = [v["label"] for v in votes]
        label_counts = Counter(labels)
        
        jury_labels = labels
        jury_confidences = [v["confidence"] for v in votes]
        reasoning = [r.get("reasoning") for r in jury_results]
        
        n_jury = len(jury_results)
        max_agreement = label_counts.most_common(1)[0][1]
        
        # UNANIMOUS
        if max_agreement == n_jury:
            return LabelResult(
                label=winning_label,
                label_type="hard",
                tier="ACCEPT",
                training_weight=1.0,
                agreement="unanimous",
                jury_labels=jury_labels,
                jury_confidences=jury_confidences,
                reasoning=reasoning,
            )
        
        # MAJORITY with adjacent disagreement
        if max_agreement >= 2:
            minority_labels = [l for l in labels if l != winning_label]
            if minority_labels:
                try:
                    max_dist = max(abs(int(winning_label) - int(m)) for m in minority_labels)
                    if max_dist <= 1:  # adjacent
                        return LabelResult(
                            label=winning_label,
                            label_type="hard",
                            tier="ACCEPT-M",
                            training_weight=0.85,
                            agreement="majority_adjacent",
                            jury_labels=jury_labels,
                            jury_confidences=jury_confidences,
                            reasoning=reasoning,
                        )
                except (ValueError, TypeError):
                    # Non-numeric labels, can't calculate distance
                    pass
        
        # Stage 4: Candidate annotation for disagreements
        if self.config.use_candidate_annotation and self.candidate_provider:
            candidate_result = await self._get_candidate_annotation(
                text, jury_results, rag_examples
            )
            
            if candidate_result and candidate_result.get("candidates"):
                candidates = candidate_result["candidates"]
                soft_label = {c["label"]: c["probability"] for c in candidates}
                primary = candidate_result.get("primary_label", winning_label)
                max_prob = max(c["probability"] for c in candidates)
                
                if max_prob >= 0.8:
                    return LabelResult(
                        label=primary,
                        label_type="hard",
                        tier="ACCEPT-M",
                        training_weight=0.8,
                        agreement="candidate_confident",
                        soft_label=soft_label,
                        jury_labels=jury_labels,
                        jury_confidences=jury_confidences,
                        reasoning=reasoning,
                    )
                else:
                    return LabelResult(
                        label=primary,
                        label_type="soft",
                        soft_label=soft_label,
                        tier="SOFT",
                        training_weight=0.7,
                        agreement="candidate_ambiguous",
                        jury_labels=jury_labels,
                        jury_confidences=jury_confidences,
                        reasoning=reasoning,
                    )
        
        # Fallback: QUARANTINE
        return LabelResult(
            label=winning_label,
            label_type="hard",
            tier="QUARANTINE",
            training_weight=0.0,
            agreement="unresolved",
            jury_labels=jury_labels,
            jury_confidences=jury_confidences,
            reasoning=reasoning,
        )
    
    async def _get_candidate_annotation(
        self,
        text: str,
        jury_results: list[dict],
        rag_examples: str
    ) -> dict[str, Any] | None:
        """Stage 4: Get candidate annotation for disagreements."""
        if not self.candidate_provider:
            return None
        
        try:
            system_prompt, user_prompt = self.prompts.build_candidate_prompt(
                text, jury_results
            )
            
            response = await self.candidate_provider.call(
                system=system_prompt,
                user=user_prompt,
                temperature=self.config.candidate_temperature,
            )
            
            return response.parsed_json
        
        except Exception as e:
            logger.warning(f"Candidate annotation failed: {e}")
            return None
    
    async def label_dataframe(
        self,
        df: pd.DataFrame,
        output_path: str | Path,
        resume: bool = False,
    ) -> pd.DataFrame:
        """Label entire dataframe with checkpointing.
        
        Parameters:
            df: Input dataframe with text column
            output_path: Path to save results
            resume: Whether to resume from existing output
            
        Returns:
            Labeled dataframe
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        text_col = self.config.text_column
        if text_col not in df.columns:
            raise ValueError(f"Text column '{text_col}' not found in dataframe")
        
        # Resume logic
        if resume and output_path.exists():
            existing_df = pd.read_csv(output_path)
            logger.info(f"Resuming from {output_path}")
            # TODO: Implement resume logic to skip already-labeled rows
        
        all_results = []
        
        for i in range(0, len(df), self.config.batch_size):
            batch_df = df.iloc[i:i + self.config.batch_size]
            
            tasks = [
                self.label_one(row[text_col])
                for _, row in batch_df.iterrows()
            ]
            
            batch_results = await asyncio.gather(*tasks)
            
            for (idx, row), result in zip(batch_df.iterrows(), batch_results):
                result_row = {
                    text_col: row[text_col],
                    "label": result.label,
                    "label_type": result.label_type,
                    "tier": result.tier,
                    "training_weight": result.training_weight,
                    "agreement": result.agreement,
                    "jury_labels": json.dumps(result.jury_labels),
                    "jury_confidences": json.dumps(result.jury_confidences),
                    "soft_label": json.dumps(result.soft_label) if result.soft_label else None,
                }
                all_results.append(result_row)
            
            # Checkpoint
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(output_path, index=False)
            
            done = min(i + self.config.batch_size, len(df))
            logger.info(
                f"Labeled {done}/{len(df)} | "
                f"ACCEPT: {sum(1 for r in all_results if r['tier']=='ACCEPT')} | "
                f"SOFT: {sum(1 for r in all_results if r['tier']=='SOFT')} | "
                f"QUARANTINE: {sum(1 for r in all_results if r['tier']=='QUARANTINE')}"
            )
        
        return pd.DataFrame(all_results)
