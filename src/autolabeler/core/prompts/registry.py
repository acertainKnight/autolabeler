"""Prompt registry for loading and assembling dataset-specific prompts."""

from pathlib import Path
from typing import Any


class PromptRegistry:
    """Loads, caches, and assembles prompt files for a given dataset.
    
    Each dataset has a directory under prompts/ with markdown files:
    - system.md: System prompt defining the LLM's role and expertise
    - rules.md: Classification rules and decision framework
    - examples.md: Calibration examples with reasoning
    - mistakes.md: Common errors to avoid
    - candidate.md (optional): Prompt for candidate annotation on disagreements
    
    Args:
        dataset: Dataset name (e.g., "fed_headlines", "tpu")
        prompts_dir: Root directory containing prompt subdirectories
        
    Example:
        >>> registry = PromptRegistry("fed_headlines")
        >>> system, user = registry.build_labeling_prompt("FED SEES RATES RISING")
        >>> print(user)  # Contains rules + examples + mistakes + text
    """
    
    def __init__(
        self,
        dataset: str,
        prompts_dir: str | Path = "prompts"
    ):
        """Initialize registry for a specific dataset.
        
        Parameters:
            dataset: Name of dataset (must have prompts/{dataset}/ directory)
            prompts_dir: Root prompts directory path
        """
        self.dataset = dataset
        self.prompts_dir = Path(prompts_dir)
        self.dataset_dir = self.prompts_dir / dataset
        
        if not self.dataset_dir.exists():
            raise ValueError(
                f"Prompt directory not found: {self.dataset_dir}. "
                f"Create prompts/{dataset}/ with required prompt files."
            )
        
        # Cache loaded prompts
        self._cache: dict[str, str] = {}
    
    def get(self, name: str) -> str:
        """Load a prompt file by name.
        
        Parameters:
            name: Prompt name without extension (e.g., "system", "rules", "examples")
            
        Returns:
            Prompt content as string
            
        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        if name in self._cache:
            return self._cache[name]
        
        prompt_path = self.dataset_dir / f"{name}.md"
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {prompt_path}. "
                f"Expected prompts/{self.dataset}/{name}.md"
            )
        
        content = prompt_path.read_text(encoding='utf-8')
        self._cache[name] = content
        return content
    
    def exists(self, name: str) -> bool:
        """Check if a prompt file exists.
        
        Parameters:
            name: Prompt name without extension
            
        Returns:
            True if file exists
        """
        return (self.dataset_dir / f"{name}.md").exists()
    
    def build_labeling_prompt(
        self,
        text: str,
        rag_examples: str = ""
    ) -> tuple[str, str]:
        """Assemble full labeling prompt from components.
        
        Combines: system + rules + examples + mistakes + RAG + text
        
        Parameters:
            text: Text to label
            rag_examples: Optional RAG-retrieved examples (formatted string)
            
        Returns:
            Tuple of (system_prompt, user_prompt)
            
        Example:
            >>> system, user = registry.build_labeling_prompt("FED SEES RATES RISING")
            >>> # system contains domain expertise
            >>> # user contains rules + examples + mistakes + text
        """
        system_prompt = self.get("system")
        
        user_parts = []
        
        # Add classification rules
        user_parts.append(self.get("rules"))
        user_parts.append("")  # blank line
        
        # Add calibration examples
        user_parts.append(self.get("examples"))
        user_parts.append("")
        
        # Add common mistakes
        user_parts.append(self.get("mistakes"))
        user_parts.append("")
        
        # Add RAG examples if provided
        if rag_examples:
            user_parts.append("## Similar Previously-Labeled Examples")
            user_parts.append(rag_examples)
            user_parts.append("")
        
        # Add the text to classify
        user_parts.append("---")
        user_parts.append("")
        user_parts.append("## CLASSIFY THIS TEXT")
        user_parts.append("")
        user_parts.append(f'"{text}"')
        user_parts.append("")
        user_parts.append("Return your classification as a JSON object with:")
        user_parts.append('{"label": <int>, "confidence": <float 0-1>, "reasoning": "<explanation>"}')
        
        user_prompt = "\n".join(user_parts)
        
        return system_prompt, user_prompt
    
    def build_candidate_prompt(
        self,
        text: str,
        jury_results: list[dict[str, Any]]
    ) -> tuple[str, str]:
        """Assemble candidate annotation prompt for disagreements.
        
        Used when jury members disagree. Asks a strong model to output
        probability distribution over plausible labels rather than forcing
        a single choice.
        
        Parameters:
            text: Text to label
            jury_results: List of jury votes with labels, confidence, reasoning
            
        Returns:
            Tuple of (system_prompt, user_prompt)
            
        Example:
            >>> jury = [
            ...     {"label": 0, "confidence": 0.8, "model": "Claude"},
            ...     {"label": 1, "confidence": 0.7, "model": "GPT-4o"}
            ... ]
            >>> system, user = registry.build_candidate_prompt("...", jury)
        """
        if not self.exists("candidate"):
            # Fallback: use regular labeling if candidate prompt doesn't exist
            return self.build_labeling_prompt(text)
        
        system_prompt = self.get("system")
        
        # Format jury disagreement
        jury_summary = ["## Jury Disagreement"]
        jury_summary.append("")
        jury_summary.append("The jury members provided these classifications:")
        jury_summary.append("")
        for vote in jury_results:
            model = vote.get("model", "Unknown")
            label = vote.get("label")
            confidence = vote.get("confidence", 0)
            reasoning = vote.get("reasoning", "")
            jury_summary.append(f"**{model}:** label={label}, confidence={confidence:.2f}")
            if reasoning:
                if isinstance(reasoning, dict):
                    reasoning = str(reasoning)
                jury_summary.append(f"  Reasoning: {reasoning[:200]}...")
            jury_summary.append("")
        
        user_parts = []
        
        # Add candidate annotation instructions
        user_parts.append(self.get("candidate"))
        user_parts.append("")
        
        # Add jury disagreement summary
        user_parts.extend(jury_summary)
        user_parts.append("")
        
        # Add the text to classify
        user_parts.append("---")
        user_parts.append("")
        user_parts.append("## TEXT TO CLASSIFY")
        user_parts.append("")
        user_parts.append(f'"{text}"')
        user_parts.append("")
        user_parts.append("Analyze why this is ambiguous and return the candidate annotation JSON.")
        
        user_prompt = "\n".join(user_parts)
        
        return system_prompt, user_prompt
    
    def __repr__(self) -> str:
        """String representation."""
        return f"PromptRegistry(dataset='{self.dataset}', dir='{self.dataset_dir}')"
