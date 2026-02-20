"""Cascaded model escalation for cost-efficient labeling.

Implements the "Trust or Escalate" approach (ICLR 2025 Oral, Jung et al.):
start with the cheapest model, accept if confident, otherwise escalate to
more expensive models. Only runs the full jury when cheaper models disagree
or lack confidence.

Cost savings come from the observation that many annotation items are "easy"
and don't require consensus from multiple expensive models. Hard items still
get the full jury treatment.

Escalation flow:
    1. Call cheapest model (cost_tier=1)
    2. If confidence >= cascade_confidence_threshold → ACCEPT (fast path)
    3. Call next cheapest model (cost_tier=2)
    4. If both agree with confidence >= cascade_agreement_threshold → ACCEPT
    5. If they disagree → escalate to remaining models (full jury)
    6. Proceed with standard aggregation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from ..dataset_config import DatasetConfig, ModelConfig


@dataclass
class EscalationResult:
    """Result of the cascade escalation process.
    
    Attributes:
        jury_results: Collected results from models called so far
        models_called: Number of models actually invoked
        total_models: Total models available in the jury
        early_exit: Whether the cascade accepted early (before full jury)
        escalation_reason: Why escalation happened (or "accepted_early")
        cost_saved_pct: Estimated cost savings vs running full jury
    """
    jury_results: list[dict[str, Any]]
    models_called: int
    total_models: int
    early_exit: bool
    escalation_reason: str
    cost_saved_pct: float


class CascadeStrategy:
    """Manages the order and gating logic for cascaded model calls.
    
    Given a list of jury models ordered by cost_tier, decides after each
    model call whether to accept or escalate.
    
    Example:
        >>> strategy = CascadeStrategy(config)
        >>> # Get ordered model indices
        >>> for tier_indices in strategy.tiers():
        ...     results = await call_models(tier_indices)
        ...     if strategy.should_accept(all_results):
        ...         break  # Early exit
    """
    
    def __init__(self, config: DatasetConfig):
        """Initialize cascade strategy.
        
        Parameters:
            config: Dataset configuration with cascade settings and jury models
        """
        self.config = config
        self.confidence_threshold = config.cascade_confidence_threshold
        self.agreement_threshold = config.cascade_agreement_threshold
        
        # Sort jury models by cost_tier (cheapest first)
        self._tier_order = self._build_tier_order(config.jury_models)
    
    def _build_tier_order(
        self,
        models: list[ModelConfig],
    ) -> list[list[int]]:
        """Group model indices by cost tier, ordered cheapest first.
        
        Parameters:
            models: List of jury model configurations
            
        Returns:
            List of lists - each inner list contains indices for one tier.
            e.g. [[2], [0, 1]] means model 2 is cheapest, models 0 and 1
            are in the next tier.
        """
        tier_map: dict[int, list[int]] = {}
        for i, m in enumerate(models):
            tier_map.setdefault(m.cost_tier, []).append(i)
        
        # Sort by tier number (ascending = cheapest first)
        return [tier_map[t] for t in sorted(tier_map.keys())]
    
    def tiers(self) -> list[list[int]]:
        """Return model indices grouped by cost tier, cheapest first.
        
        Returns:
            List of index groups, e.g. [[2], [0, 1]]
        """
        return self._tier_order
    
    def should_accept(
        self,
        results: list[dict[str, Any]],
        tier_index: int,
    ) -> tuple[bool, str]:
        """Decide whether to accept current results or escalate.
        
        Parameters:
            results: All jury results collected so far
            tier_index: Which tier we just finished (0-indexed)
            
        Returns:
            Tuple of (should_accept, reason)
        """
        if not results:
            return False, "no_results"
        
        # Filter to valid results (have a label)
        valid = [r for r in results if r.get("label") is not None]
        if not valid:
            return False, "no_valid_labels"
        
        n_valid = len(valid)
        
        # --- Tier 0 (cheapest model): single-model confidence gate ---
        if tier_index == 0 and n_valid == 1:
            conf = valid[0].get("confidence", 0.0)
            if conf >= self.confidence_threshold:
                logger.info(
                    f"Cascade: early accept from {valid[0].get('model_name', '?')} "
                    f"(confidence={conf:.2f} >= {self.confidence_threshold})"
                )
                return True, "single_model_confident"
            return False, f"single_model_low_confidence ({conf:.2f})"
        
        # --- Tier 1+ (two or more models): agreement gate ---
        if n_valid >= 2:
            # Check if all models so far agree on the same label
            labels = [r["label"] for r in valid]
            label_set = set(labels)
            
            if len(label_set) == 1:
                # Unanimous among models called so far
                avg_conf = sum(r.get("confidence", 0.5) for r in valid) / n_valid
                if avg_conf >= self.agreement_threshold:
                    logger.info(
                        f"Cascade: early accept with {n_valid}-model agreement "
                        f"on '{labels[0]}' (avg_conf={avg_conf:.2f} >= "
                        f"{self.agreement_threshold})"
                    )
                    return True, f"{n_valid}_model_agreement"
                return False, f"agreement_low_confidence ({avg_conf:.2f})"
            
            # Models disagree - check if majority is strong enough
            from collections import Counter
            counts = Counter(labels)
            top_label, top_count = counts.most_common(1)[0]
            majority_ratio = top_count / n_valid
            
            if majority_ratio >= 0.75 and n_valid >= 3:
                # Strong supermajority with enough models
                majority_confs = [
                    r.get("confidence", 0.5)
                    for r in valid if r["label"] == top_label
                ]
                avg_majority_conf = sum(majority_confs) / len(majority_confs)
                
                if avg_majority_conf >= self.agreement_threshold:
                    logger.info(
                        f"Cascade: accept with {top_count}/{n_valid} supermajority "
                        f"on '{top_label}' (avg_conf={avg_majority_conf:.2f})"
                    )
                    return True, f"supermajority_{top_count}_of_{n_valid}"
            
            return False, f"disagreement ({dict(counts)})"
        
        return False, "insufficient_models"
    
    def build_escalation_result(
        self,
        results: list[dict[str, Any]],
        models_called: int,
        early_exit: bool,
        reason: str,
    ) -> EscalationResult:
        """Package the cascade outcome into an EscalationResult.
        
        Parameters:
            results: All collected jury results
            models_called: How many models were actually called
            early_exit: Whether cascade accepted early
            reason: Reason for the decision
            
        Returns:
            EscalationResult with cost savings estimate
        """
        total = len(self.config.jury_models)
        cost_saved_pct = (
            (1.0 - models_called / total) * 100 if total > 0 else 0.0
        )
        
        return EscalationResult(
            jury_results=results,
            models_called=models_called,
            total_models=total,
            early_exit=early_exit,
            escalation_reason=reason,
            cost_saved_pct=cost_saved_pct,
        )
