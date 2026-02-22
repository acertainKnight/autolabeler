"""Unified labeling pipeline for any dataset.

This is the core of the autolabeler service. One pipeline handles all datasets,
configured by DatasetConfig and PromptRegistry.

Architecture:
1. Optional relevancy gate (cheap pre-filter)
2. Jury voting -- either full parallel jury or cascaded escalation
3. Confidence-weighted aggregation
4. Optional candidate annotation (for disagreements)
5. Optional cross-verification (for uncertain labels)
6. Tier assignment (ACCEPT, ACCEPT-M, SOFT, QUARANTINE)

Cascade mode (use_cascade=True):
  Calls cheapest model first; if confident, skips the rest.
  Otherwise escalates tier-by-tier until confidence/agreement is reached
  or all models have been called. Typically saves 40-60% of API cost.
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
from autolabeler.core.llm_providers.providers import (
    CostTracker,
    LLMProvider,
    LLMResponse,
    get_provider,
)
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
        agreement: Agreement type (unanimous, majority_adjacent, cascade_single, etc.)
        jury_labels: List of labels from each jury member
        jury_confidences: List of confidences from each jury member
        soft_label: Optional soft label distribution {label: prob}
        reasoning: Reasoning from jury members
        error: Optional error message
        cascade_info: Metadata from cascade escalation (models_called, cost_saved_pct, etc.)
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
    cascade_info: dict[str, Any] | None = None


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
        max_budget: float = 0.0,
        custom_providers: dict[str, LLMProvider] | None = None,
    ):
        """Initialize pipeline.

        Parameters:
            dataset_config: Configuration for this dataset.
            prompt_registry: Prompt registry for this dataset.
            confidence_scorer: Optional pre-calibrated confidence scorer.
            max_budget: Maximum USD spend (0 = unlimited). When exceeded the
                pipeline stops labeling and returns results collected so far.
            custom_providers: Optional mapping of model name to a pre-built
                ``LLMProvider`` instance. Keys must match the ``name`` field in
                ``ModelConfig``. When a model name is found here the pre-built
                instance is used directly instead of calling ``get_provider()``.
                This is the programmatic injection path for corporate proxies or
                any provider that requires custom constructor arguments.

                Example::

                    custom_providers = {
                        "Corp LLM": MyCorporateProxy(model="internal-v2", token="..."),
                    }
                    pipeline = LabelingPipeline(config, registry, custom_providers=custom_providers)
        """
        self.config = dataset_config
        self.prompts = prompt_registry
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()
        self.cost_tracker = CostTracker(budget=max_budget)
        self._custom_providers: dict[str, LLMProvider] = custom_providers or {}

        # Load jury weights if provided
        self.jury_weights = None
        if self.config.jury_weights_path:
            from ..quality.jury_weighting import JuryWeightLearner
            self.jury_weights = JuryWeightLearner.load(self.config.jury_weights_path)
            logger.info(f"Loaded jury weights from {self.config.jury_weights_path}")

        # Initialize providers
        self.jury_providers = [
            self._resolve_provider(m)
            for m in self.config.jury_models
        ]

        self.gate_provider = None
        if self.config.gate_model:
            self.gate_provider = self._resolve_provider(self.config.gate_model)

        self.candidate_provider = None
        if self.config.candidate_model:
            self.candidate_provider = self._resolve_provider(self.config.candidate_model)

        # Initialize cascade strategy
        self.cascade = None
        if self.config.use_cascade:
            from .cascade import CascadeStrategy
            self.cascade = CascadeStrategy(self.config)
            tier_summary = [
                f"tier {i}: {len(t)} model(s)"
                for i, t in enumerate(self.cascade.tiers())
            ]
            logger.info(f"Cascade mode enabled: {', '.join(tier_summary)}")

        # Initialize verification
        self.verifier = None
        if self.config.use_cross_verification and self.config.verification_model:
            from .verification import CrossVerifier
            verifier_provider = self._resolve_provider(self.config.verification_model)
            self.verifier = CrossVerifier(
                verifier_provider,
                self.config.verification_model,
                self.prompts,
                self.config
            )
            logger.info(f"Initialized cross-verifier with {self.config.verification_model.name}")

        logger.info(
            f"Initialized LabelingPipeline for {self.config.name} with "
            f"{len(self.jury_providers)} jury models"
        )

    def _resolve_provider(self, model_config: ModelConfig) -> LLMProvider:
        """Resolve the LLM provider for a given model config.

        Checks ``custom_providers`` by model name first, then falls back to
        ``get_provider()`` using the provider/model strings from the config.

        Parameters:
            model_config: Model configuration specifying provider, model, and name.

        Returns:
            An ``LLMProvider`` instance ready to call.
        """
        if model_config.name in self._custom_providers:
            logger.debug(f"Using injected custom provider for model '{model_config.name}'")
            return self._custom_providers[model_config.name]
        return get_provider(model_config.provider, model_config.model)
    
    async def label_one(self, text: str, rag_examples: str = "") -> LabelResult:
        """Label a single text through the pipeline.
        
        When cascade mode is enabled, starts with the cheapest model and
        escalates only if confidence is low. Otherwise runs the full jury
        in parallel.
        
        Parameters:
            text: Text to label
            rag_examples: Optional RAG-retrieved examples
            
        Returns:
            LabelResult with label, confidence, tier, etc.
        """
        try:
            # Stage 1: Optional relevancy gate (skip for now - jury handles it)
            
            # Stage 2: Jury voting (cascade or full parallel)
            system_prompt, user_prompt = self.prompts.build_labeling_prompt(
                text, rag_examples
            )
            
            if self.cascade:
                jury_results, escalation = await self._call_jury_cascade(
                    system_prompt, user_prompt
                )
            else:
                jury_results = await self._call_jury(system_prompt, user_prompt)
                escalation = None
            
            # Need at least 1 result (cascade can accept a single confident model)
            min_results = 1 if self.cascade else 2
            if len(jury_results) < min_results:
                return LabelResult(
                    label=None,
                    label_type="hard",
                    tier="QUARANTINE",
                    training_weight=0.0,
                    agreement="jury_failure",
                    jury_labels=[],
                    jury_confidences=[],
                    error=f"Jury failed (< {min_results} models responded)"
                )
            
            # Stage 3: Confidence-weighted aggregation
            result = await self._aggregate_votes(text, jury_results, rag_examples)
            
            # Attach cascade metadata if available
            if escalation:
                result.cascade_info = {
                    "models_called": escalation.models_called,
                    "total_models": escalation.total_models,
                    "early_exit": escalation.early_exit,
                    "reason": escalation.escalation_reason,
                    "cost_saved_pct": escalation.cost_saved_pct,
                }
            
            return result
        
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
    
    def _compute_juror_confidence(self, result: dict[str, Any]) -> float:
        """Compute best-available confidence for a single juror result.
        
        Uses logprobs if available (most reliable), then calibrates.
        Falls back to verbal confidence only as a last resort.
        
        Parameters:
            result: Raw juror result dict from _call_one_juror
            
        Returns:
            Calibrated confidence score (0-1)
        """
        model_config = result["_model_config"]
        label = result["label"]
        
        if model_config.has_logprobs and result.get("logprobs"):
            raw_conf = self.confidence_scorer.from_logprobs(
                result["logprobs"], label
            )
        else:
            raw_conf = result.get("confidence", 0.7)
        
        return self.confidence_scorer.calibrate(raw_conf)
    
    async def _call_jury_cascade(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[list[dict[str, Any]], Any]:
        """Call jury models using cascaded escalation (cheapest first).
        
        Calls models tier by tier. After each tier, computes calibrated
        confidence (using logprobs where available) and checks whether
        that confidence is high enough to accept early.
        
        Returns:
            Tuple of (jury_results, EscalationResult)
        """
        all_results: list[dict[str, Any]] = []
        models_called = 0
        
        for tier_idx, model_indices in enumerate(self.cascade.tiers()):
            # Call all models in this tier in parallel
            tasks = []
            for idx in model_indices:
                provider = self.jury_providers[idx]
                model_config = self.config.jury_models[idx]
                tasks.append(
                    self._call_one_juror(
                        provider, model_config, system_prompt, user_prompt
                    )
                )
            
            tier_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, r in enumerate(tier_results):
                model_idx = model_indices[i]
                if isinstance(r, Exception):
                    logger.warning(f"Cascade juror {model_idx} failed: {r}")
                    continue
                r["_model_config"] = self.config.jury_models[model_idx]
                # Replace verbal confidence with calibrated confidence
                # BEFORE the cascade gate checks it
                r["confidence"] = self._compute_juror_confidence(r)
                all_results.append(r)
                models_called += 1
            
            # Check if we can accept early
            should_accept, reason = self.cascade.should_accept(all_results, tier_idx)
            
            if should_accept:
                escalation = self.cascade.build_escalation_result(
                    results=all_results,
                    models_called=models_called,
                    early_exit=True,
                    reason=reason,
                )
                logger.info(
                    f"Cascade: accepted at tier {tier_idx} "
                    f"({models_called}/{len(self.config.jury_models)} models, "
                    f"saved ~{escalation.cost_saved_pct:.0f}% cost) — {reason}"
                )
                return all_results, escalation
        
        # All tiers exhausted — fall through to full aggregation
        escalation = self.cascade.build_escalation_result(
            results=all_results,
            models_called=models_called,
            early_exit=False,
            reason="full_jury_required",
        )
        logger.debug(f"Cascade: full jury used ({models_called} models)")
        return all_results, escalation
    
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
        # Build JSON schema if structured output is enabled
        response_schema = None
        if self.config.use_structured_output:
            response_schema = {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "enum": self.config.labels,
                        "description": "The predicted label"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation for the label choice"
                    },
                    "confidence": {
                        "type": "number",
                        "description": (
                            "Your confidence in this label as a probability "
                            "between 0 and 1. Be honest: 0.5 means a coin flip, "
                            "0.9 means very confident."
                        ),
                    }
                },
                "required": ["label", "confidence", "reasoning"],
                "additionalProperties": False
            }
        
        response: LLMResponse = await provider.call(
            system=system_prompt,
            user=user_prompt,
            temperature=self.config.jury_temperature,
            logprobs=model_config.has_logprobs,
            response_schema=response_schema,
        )
        
        self.cost_tracker.add(response.cost)
        
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
            raw_conf = response.parsed_json.get("confidence")
            if isinstance(raw_conf, (int, float)):
                result["confidence"] = max(0.0, min(1.0, float(raw_conf)))
            elif isinstance(raw_conf, str):
                result["confidence"] = self.confidence_scorer.from_verbal(raw_conf)
        
        return result
    
    async def _aggregate_votes(
        self,
        text: str,
        jury_results: list[dict],
        rag_examples: str
    ) -> LabelResult:
        """Stage 3: Aggregate jury votes with confidence weighting.
        
        Also handles stages 4-5: candidate annotation and tier assignment.
        Handles single-model cascade accepts as well as full-jury results.
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
            
            # Apply jury weights if available
            if self.jury_weights:
                weight = self.jury_weights.get_weight(model_config.name, label)
                conf = conf * weight
            
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
        
        # ALWAYS compute confidence-weighted soft label distribution
        total_weight = sum(label_weights.values())
        soft_label = {
            label: weight / total_weight
            for label, weight in label_weights.items()
        }
        
        n_jury = len(jury_results)
        max_agreement = label_counts.most_common(1)[0][1]
        
        # SINGLE-MODEL CASCADE ACCEPT
        if n_jury == 1 and self.cascade:
            conf = jury_confidences[0]
            tier = "ACCEPT" if conf >= self.config.cascade_confidence_threshold else "SOFT"
            return LabelResult(
                label=winning_label,
                label_type="hard",
                tier=tier,
                training_weight=min(conf, 0.95),
                agreement="cascade_single",
                jury_labels=jury_labels,
                jury_confidences=jury_confidences,
                soft_label=soft_label,
                reasoning=reasoning,
            )
        
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
                soft_label=soft_label,
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
                            soft_label=soft_label,
                            reasoning=reasoning,
                        )
                except (ValueError, TypeError):
                    # Non-numeric labels, can't calculate distance
                    pass
        
        # Stage 4: Candidate annotation for disagreements
        candidate_soft_label = None
        if self.config.use_candidate_annotation and self.candidate_provider:
            candidate_result = await self._get_candidate_annotation(
                text, jury_results, rag_examples
            )
            
            if candidate_result and candidate_result.get("candidates"):
                candidates = candidate_result["candidates"]
                candidate_soft_label = {c["label"]: c["probability"] for c in candidates}
                primary = candidate_result.get("primary_label", winning_label)
                max_prob = max(c["probability"] for c in candidates)
                
                # Update winning label if candidate is more confident
                if max_prob >= 0.8:
                    winning_label = primary
                    soft_label = candidate_soft_label
        
        # Stage 5: Cross-verification for uncertain labels
        verified = False
        avg_confidence = sum(jury_confidences) / len(jury_confidences) if jury_confidences else 0.5
        jury_agreement = max_agreement / n_jury
        
        should_verify = (
            self.verifier and
            (jury_agreement < self.config.verification_threshold or
             avg_confidence < self.config.verification_threshold)
        )
        
        if should_verify:
            verification_result = await self.verifier.verify(
                text=text,
                proposed_label=winning_label,
                jury_votes=label_counts,
                confidence=avg_confidence,
                reasoning=reasoning,
            )
            
            if verification_result["action"] == "override":
                logger.info(
                    f"Verification override: {winning_label} -> "
                    f"{verification_result.get('corrected_label')}"
                )
                winning_label = verification_result.get("corrected_label", winning_label)
                # Update soft label to reflect verified decision
                soft_label = {winning_label: 1.0}
                verified = True
        
        # Final tier assignment based on verification and candidate results
        if candidate_soft_label and max_prob >= 0.8:
            return LabelResult(
                label=winning_label,
                label_type="hard",
                tier="ACCEPT" if verified else "ACCEPT-M",
                training_weight=1.0 if verified else 0.8,
                agreement="verified" if verified else "candidate_confident",
                soft_label=soft_label,
                jury_labels=jury_labels,
                jury_confidences=jury_confidences,
                reasoning=reasoning,
            )
        elif candidate_soft_label:
            return LabelResult(
                label=winning_label,
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
            soft_label=soft_label,
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
        
        # Resume: reload previous checkpoint and skip already-labeled rows
        all_results = []
        if resume and output_path.exists():
            existing_df = pd.read_csv(output_path)
            if text_col in existing_df.columns:
                already_labeled = set(existing_df[text_col].astype(str))
                original_len = len(df)
                df = df[~df[text_col].astype(str).isin(already_labeled)].reset_index(drop=True)
                # Carry forward previous results so the final CSV is complete
                all_results = existing_df.to_dict("records")
                logger.info(
                    f"Resumed from {output_path}: "
                    f"{len(already_labeled)} already labeled, "
                    f"{len(df)} remaining of {original_len} total"
                )
            else:
                logger.warning(
                    f"Checkpoint file missing '{text_col}' column -- "
                    f"starting from scratch"
                )
        
        for i in range(0, len(df), self.config.batch_size):
            # Budget gate: stop before the next batch if budget is exhausted
            if self.cost_tracker.budget_exceeded:
                logger.warning(
                    f"Budget exhausted (${self.cost_tracker.total_cost:.4f} "
                    f"of ${self.cost_tracker.budget:.2f}) -- "
                    f"stopping after {len(all_results)} labeled samples"
                )
                break
            
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
                if result.cascade_info:
                    result_row["cascade_models_called"] = result.cascade_info["models_called"]
                    result_row["cascade_early_exit"] = result.cascade_info["early_exit"]
                all_results.append(result_row)
            
            # Checkpoint
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(output_path, index=False)
            
            done = min(i + self.config.batch_size, len(df))
            cost_str = f"${self.cost_tracker.total_cost:.4f}"
            if self.cost_tracker.budget > 0:
                cost_str += f" / ${self.cost_tracker.budget:.2f}"
            status = (
                f"Labeled {done}/{len(df)} | "
                f"cost: {cost_str} | "
                f"ACCEPT: {sum(1 for r in all_results if r['tier']=='ACCEPT')} | "
                f"SOFT: {sum(1 for r in all_results if r['tier']=='SOFT')} | "
                f"QUARANTINE: {sum(1 for r in all_results if r['tier']=='QUARANTINE')}"
            )
            if self.cascade:
                early_exits = sum(
                    1 for r in all_results if r.get("cascade_early_exit", False)
                )
                status += f" | cascade early exits: {early_exits}/{done}"
            logger.info(status)
        
        final_df = pd.DataFrame(all_results)

        # Post-labeling diagnostics hook
        if (
            self.config.diagnostics
            and self.config.diagnostics.enabled
            and self.config.diagnostics.run_post_labeling
        ):
            try:
                from ..diagnostics import run_diagnostics
                diagnostics_dir = output_path.parent / 'diagnostics'
                logger.info(f'Running post-labeling diagnostics -> {diagnostics_dir}')
                run_diagnostics(
                    labeled_df=final_df,
                    config=self.config,
                    output_dir=diagnostics_dir,
                )
            except Exception as e:
                logger.error(f'Post-labeling diagnostics failed (non-fatal): {e}')

        return final_df
