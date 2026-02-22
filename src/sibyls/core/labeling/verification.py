"""Cross-verification stage for uncertain labels.

Implements the +58% Cohen's kappa improvement from verification-oriented orchestration
(arXiv 2511.09785). Cross-family verification (e.g., Claude verifying GPT's label)
outperforms same-family verification.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from ..dataset_config import DatasetConfig, ModelConfig
from ..llm_providers import LLMProvider
from ..prompts.registry import PromptRegistry


class CrossVerifier:
    """Verifies uncertain labels using a different model family.
    
    Triggers when:
    - jury_agreement < threshold (e.g., < 0.67 for 3 jurors)
    - OR calibrated_confidence < confidence_threshold (e.g., < 0.6)
    
    The verifier receives: the text, the proposed label, the jury vote 
    distribution, and any reasoning. It outputs: confirm/override + 
    optional corrected label.
    
    Cross-family verification (Claude checking Gemini's work) significantly
    outperforms same-family verification.
    
    Attributes:
        verifier_provider: LLM provider for verification
        verifier_model: Model config for verifier
        prompts: Prompt registry for building verification prompts
        config: Dataset configuration
        
    Example:
        >>> verifier = CrossVerifier(provider, model_config, prompts, config)
        >>> result = await verifier.verify(
        ...     text="Fed signals patience on rates",
        ...     proposed_label="0",
        ...     jury_votes={"0": 2, "-1": 1},
        ...     confidence=0.55
        ... )
        >>> if result["action"] == "override":
        ...     print(f"Corrected to {result['corrected_label']}")
    """
    
    def __init__(
        self,
        verifier_provider: LLMProvider,
        verifier_model: ModelConfig,
        prompts: PromptRegistry,
        config: DatasetConfig,
    ):
        """Initialize cross-verifier.
        
        Parameters:
            verifier_provider: LLM provider for verification
            verifier_model: Model config for verifier
            prompts: Prompt registry
            config: Dataset configuration
        """
        self.verifier_provider = verifier_provider
        self.verifier_model = verifier_model
        self.prompts = prompts
        self.config = config
        
    async def verify(
        self,
        text: str,
        proposed_label: str,
        jury_votes: dict[str, int],
        confidence: float,
        reasoning: list[str] | None = None,
    ) -> dict[str, Any]:
        """Verify an uncertain label.
        
        Parameters:
            text: Input text to classify
            proposed_label: Label proposed by jury
            jury_votes: Vote distribution (label -> count)
            confidence: Calibrated confidence in proposed label
            reasoning: Optional reasoning from jury members
            
        Returns:
            Dict with:
                - action: "confirm" or "override"
                - confidence: Verifier's confidence
                - corrected_label: If overriding, the corrected label
                - reasoning: Verifier's reasoning
        """
        # Build verification prompt
        system_prompt, user_prompt = self._build_verification_prompt(
            text, proposed_label, jury_votes, confidence, reasoning
        )
        
        # Build response schema if structured output is enabled
        response_schema = None
        if self.config.use_structured_output:
            response_schema = {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["confirm", "override"],
                        "description": "Whether to confirm or override the jury's label"
                    },
                    "corrected_label": {
                        "type": "string",
                        "enum": self.config.labels,
                        "description": "If overriding, the corrected label"
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Confidence in this verification decision"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation for the decision"
                    }
                },
                "required": ["action", "reasoning"],
                "additionalProperties": False
            }
        
        try:
            response = await self.verifier_provider.call(
                system=system_prompt,
                user=user_prompt,
                temperature=0.1,  # Low temperature for verification
                response_schema=response_schema,
            )
            
            if not response.parsed_json:
                logger.warning("Verification failed to parse JSON response")
                return {"action": "confirm", "confidence": confidence}
            
            result = response.parsed_json
            
            # Ensure required fields
            if "action" not in result:
                result["action"] = "confirm"
            
            # If overriding, must have corrected_label
            if result["action"] == "override" and "corrected_label" not in result:
                logger.warning("Override without corrected_label, falling back to confirm")
                result["action"] = "confirm"
            
            return result
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {"action": "confirm", "confidence": confidence}
    
    def _build_verification_prompt(
        self,
        text: str,
        proposed_label: str,
        jury_votes: dict[str, int],
        confidence: float,
        reasoning: list[str] | None = None,
    ) -> tuple[str, str]:
        """Build verification prompt.
        
        The verifier sees the full context: text, proposed label, vote distribution,
        and jury reasoning. Its job is to catch errors, not to re-label from scratch.
        """
        # Load verification prompt template if it exists, otherwise use a default
        if self.prompts.exists("verify"):
            verify_template = self.prompts.get("verify")
        else:
            verify_template = self._default_verification_template()
        
        # Build system prompt
        system_prompt = verify_template
        
        # Build user prompt with full context
        vote_str = ", ".join(f"{label}: {count} vote(s)" for label, count in sorted(jury_votes.items()))
        reasoning_str = ""
        if reasoning:
            reasoning_filtered = [r for r in reasoning if r]
            if reasoning_filtered:
                reasoning_str = "\n\nJury reasoning:\n" + "\n".join(f"- {r}" for r in reasoning_filtered)
        
        user_prompt = f"""TEXT TO VERIFY:
{text}

JURY DECISION:
- Proposed label: {proposed_label}
- Vote distribution: {vote_str}
- Confidence: {confidence:.2f}
{reasoning_str}

TASK:
Review the jury's decision. As an independent verifier, decide whether to:
1. CONFIRM: The jury's label is correct
2. OVERRIDE: The jury made an error, provide the corrected label

Respond in JSON format with:
- action: "confirm" or "override"
- corrected_label: (only if overriding) the correct label
- confidence: "low", "medium", or "high"
- reasoning: brief explanation (1-2 sentences)
"""
        
        return system_prompt, user_prompt
    
    def _default_verification_template(self) -> str:
        """Default verification system prompt if verify.md doesn't exist."""
        return f"""You are an expert verifier for text classification.

Your task is to review jury decisions and catch errors. You will receive:
1. The text to classify
2. The jury's proposed label
3. How the jury members voted
4. The jury's confidence and reasoning

Valid labels: {', '.join(self.config.labels)}

As a verifier, you should:
- Be conservative: only override if you're confident the jury made an error
- Focus on catching clear mistakes, not debating borderline cases
- Consider the vote distribution: unanimous decisions are less likely to be wrong
- Apply the same classification rules as the jury

Cross-check the text against the proposed label. If it's clearly wrong, override it.
If it's reasonable or you're uncertain, confirm it."""
