"""ALCHEmist-style program generation for hybrid program/jury labeling.

This module implements the ALCHEmist approach where a few-shot LLM generates
candidate labeling programs (Python functions) that are then evaluated on
high-confidence jury-labeled data to identify useful heuristics.

Evidence base:
- ALCHEmist (arXiv 2410.13089): generates Python functions as labeling programs,
  filters by precision/recall, then uses them alongside jury for improved coverage
- Snorkel-style weak supervision: combine multiple noisy signals via generative model
- Hybrid approach: program-based heuristics for scalability + jury for quality

Workflow:
1. Select high-confidence jury-labeled examples as "seed" data
2. Prompt few-shot LLM to generate Python labeling functions
3. Execute functions safely on seed data, measure precision/recall/coverage
4. Keep only high-quality programs (precision > threshold)
5. For new data: run programs first, fallback to jury for uncertain cases
"""

from __future__ import annotations

import ast
import json
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from loguru import logger

from ..llm_providers.providers import LLMProvider
from ..dataset_config import DatasetConfig, ModelConfig
from ..prompts.registry import PromptRegistry


class ProgramGenerator:
    """Generates candidate labeling programs using few-shot LLM prompting.
    
    Generates Python functions that take text as input and return a label.
    Programs are inspired by the ALCHEmist approach.
    
    Example:
        >>> generator = ProgramGenerator(provider, model, prompts, config)
        >>> programs = generator.generate(seed_df, n_programs=10)
        >>> # Returns list of {"code": "...", "description": "..."}
    """
    
    def __init__(
        self,
        provider: LLMProvider,
        model: ModelConfig,
        prompts: PromptRegistry,
        config: DatasetConfig,
    ):
        """Initialize program generator.
        
        Parameters:
            provider: LLM provider for program generation
            model: Model configuration
            prompts: Prompt registry (expects 'program_gen.md')
            config: Dataset configuration
        """
        self.provider = provider
        self.model = model
        self.prompts = prompts
        self.config = config
    
    async def generate(
        self,
        seed_df: pd.DataFrame,
        n_programs: int = 10,
        max_examples_per_class: int = 5,
    ) -> list[dict[str, Any]]:
        """Generate candidate labeling programs.
        
        Parameters:
            seed_df: High-confidence labeled examples (must have 'text' and 'label')
            n_programs: Number of programs to generate
            max_examples_per_class: Max examples per label class to include in prompt
            
        Returns:
            List of program dicts with 'code' and 'description'
        """
        # Build few-shot examples (balanced across labels)
        examples = self._sample_balanced_examples(seed_df, max_examples_per_class)
        
        # Build prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(examples, n_programs)
        
        logger.info(f"Generating {n_programs} candidate programs with {len(examples)} seed examples")
        
        # Call LLM
        response = await self.provider.call(
            model=self.model.name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.model.temperature,
            max_tokens=self.model.max_tokens or 4096,
        )
        
        # Parse programs from response
        programs = self._parse_programs(response.content)
        
        logger.info(f"Generated {len(programs)} programs successfully")
        
        return programs
    
    def _sample_balanced_examples(
        self,
        df: pd.DataFrame,
        max_per_class: int,
    ) -> list[dict[str, str]]:
        """Sample balanced examples across label classes.
        
        Parameters:
            df: DataFrame with 'text' and 'label' columns
            max_per_class: Max examples per label
            
        Returns:
            List of {"text": ..., "label": ...}
        """
        examples = []
        for label in df["label"].unique():
            label_df = df[df["label"] == label].sample(
                n=min(max_per_class, len(df[df["label"] == label])),
                random_state=42,
            )
            for _, row in label_df.iterrows():
                examples.append({
                    "text": row["text"],
                    "label": str(row["label"]),
                })
        
        return examples
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for program generation."""
        # Try to load from prompts registry
        try:
            return self.prompts.get_prompt("program_gen", "system")
        except (KeyError, FileNotFoundError):
            # Fallback default
            return self._default_system_prompt()
    
    def _build_user_prompt(
        self,
        examples: list[dict[str, str]],
        n_programs: int,
    ) -> str:
        """Build user prompt with examples and request.
        
        Parameters:
            examples: List of seed examples
            n_programs: Number of programs to generate
            
        Returns:
            Formatted user prompt
        """
        # Format examples
        examples_text = "\n".join([
            f"Text: {ex['text']}\nLabel: {ex['label']}\n"
            for ex in examples
        ])
        
        label_desc = ", ".join([f"{k}: {v}" for k, v in self.config.labels.items()])
        
        return f"""Generate {n_programs} diverse Python labeling functions.

Valid labels: {label_desc}

Here are some labeled examples to guide you:

{examples_text}

For each program, return:
1. A Python function named 'label_fn(text: str) -> str' that returns a label
2. A brief description of the heuristic

Output as JSON array:
[
  {{"code": "def label_fn(text: str) -> str:\\n    ...", "description": "..."}},
  ...
]
"""
    
    def _parse_programs(self, response_text: str) -> list[dict[str, Any]]:
        """Parse generated programs from LLM response.
        
        Parameters:
            response_text: Raw LLM response
            
        Returns:
            List of program dicts
        """
        # Try to extract JSON array
        try:
            # Look for JSON array in response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.warning("No JSON array found in response")
                return []
            
            json_text = response_text[start_idx:end_idx]
            programs = json.loads(json_text)
            
            # Validate each program
            valid_programs = []
            for prog in programs:
                if "code" in prog and "description" in prog:
                    valid_programs.append(prog)
                else:
                    logger.warning("Skipping invalid program (missing code or description)")
            
            return valid_programs
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse programs as JSON: {e}")
            return []
    
    def _default_system_prompt(self) -> str:
        """Default system prompt for program generation."""
        return """You are an expert at writing Python labeling functions for text classification.

Your task is to generate diverse heuristic functions that can accurately label text examples.
Each function should:
1. Take a single string argument (text) and return a string label
2. Implement a clear, interpretable heuristic (keyword matching, pattern detection, etc.)
3. Be safe to execute (no file I/O, network calls, or dangerous operations)
4. Focus on ONE specific pattern or heuristic per function

Good heuristics include:
- Keyword presence/absence
- Phrase patterns
- Sentiment indicators
- Named entity presence
- Text length or structure
- Negation detection

Output your functions as valid JSON with 'code' and 'description' fields."""


class ProgramLabeler:
    """Evaluates and applies labeling programs to data.
    
    Measures program quality (precision, recall, coverage) on seed data,
    then applies high-quality programs to new data as a first-pass filter.
    
    Example:
        >>> labeler = ProgramLabeler()
        >>> labeler.evaluate_programs(programs, seed_df)
        >>> labels = labeler.apply_programs(test_df)
    """
    
    def __init__(
        self,
        precision_threshold: float = 0.8,
        coverage_threshold: float = 0.1,
    ):
        """Initialize program labeler.
        
        Parameters:
            precision_threshold: Minimum precision to keep a program
            coverage_threshold: Minimum coverage (fraction abstaining) to keep
        """
        self.precision_threshold = precision_threshold
        self.coverage_threshold = coverage_threshold
        self.programs: list[dict[str, Any]] = []
        self.program_stats: list[dict[str, Any]] = []
    
    def evaluate_programs(
        self,
        programs: list[dict[str, Any]],
        seed_df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Evaluate programs on seed data and filter by quality.
        
        Parameters:
            programs: List of program dicts with 'code' and 'description'
            seed_df: Labeled seed data (must have 'text' and 'label')
            
        Returns:
            List of program stats (precision, recall, coverage)
        """
        logger.info(f"Evaluating {len(programs)} programs on {len(seed_df)} seed examples")
        
        stats = []
        valid_programs = []
        
        for i, program in enumerate(programs):
            try:
                # Compile and run program
                label_fn = self._compile_program(program["code"])
                predictions = []
                
                for text in seed_df["text"]:
                    try:
                        pred = label_fn(text)
                        predictions.append(str(pred) if pred is not None else None)
                    except Exception as e:
                        logger.debug(f"Program {i} failed on example: {e}")
                        predictions.append(None)
                
                # Compute metrics
                seed_df = seed_df.copy()
                seed_df["pred"] = predictions
                
                # Coverage: fraction that didn't abstain (return None)
                coverage = (seed_df["pred"].notna()).sum() / len(seed_df)
                
                # Precision/recall on non-abstaining predictions
                covered = seed_df[seed_df["pred"].notna()]
                if len(covered) > 0:
                    correct = (covered["pred"] == covered["label"].astype(str)).sum()
                    precision = correct / len(covered)
                    
                    # Recall: fraction of all true labels that were correctly predicted
                    recall = correct / len(seed_df)
                else:
                    precision = 0.0
                    recall = 0.0
                
                program_stat = {
                    "program_id": i,
                    "description": program["description"],
                    "precision": precision,
                    "recall": recall,
                    "coverage": coverage,
                    "n_covered": len(covered),
                }
                
                stats.append(program_stat)
                
                # Keep if quality thresholds met
                if precision >= self.precision_threshold and coverage >= self.coverage_threshold:
                    valid_programs.append({**program, "stats": program_stat})
                    logger.info(
                        f"Program {i} KEPT: precision={precision:.2f}, "
                        f"recall={recall:.2f}, coverage={coverage:.2f}"
                    )
                else:
                    logger.info(
                        f"Program {i} REJECTED: precision={precision:.2f}, "
                        f"coverage={coverage:.2f}"
                    )
            
            except Exception as e:
                logger.warning(f"Program {i} failed to compile or run: {e}")
                stats.append({
                    "program_id": i,
                    "description": program.get("description", "N/A"),
                    "error": str(e),
                })
        
        self.programs = valid_programs
        self.program_stats = stats
        
        logger.info(f"Kept {len(valid_programs)} / {len(programs)} programs")
        
        return stats
    
    def apply_programs(
        self,
        df: pd.DataFrame,
        return_confidence: bool = True,
    ) -> pd.DataFrame:
        """Apply validated programs to data.
        
        Parameters:
            df: DataFrame with 'text' column
            return_confidence: If True, compute confidence from agreement
            
        Returns:
            DataFrame with 'program_label' and optionally 'program_confidence'
        """
        if not self.programs:
            raise ValueError("No programs available. Run evaluate_programs() first.")
        
        logger.info(f"Applying {len(self.programs)} programs to {len(df)} examples")
        
        # Run each program
        all_predictions = []
        for program in self.programs:
            label_fn = self._compile_program(program["code"])
            preds = []
            for text in df["text"]:
                try:
                    pred = label_fn(text)
                    preds.append(str(pred) if pred is not None else None)
                except Exception:
                    preds.append(None)
            all_predictions.append(preds)
        
        # Aggregate via majority vote
        df = df.copy()
        labels = []
        confidences = []
        
        for i in range(len(df)):
            votes = [p[i] for p in all_predictions if p[i] is not None]
            
            if not votes:
                # All programs abstained
                labels.append(None)
                confidences.append(0.0)
            else:
                # Majority vote
                vote_counts = {}
                for v in votes:
                    vote_counts[v] = vote_counts.get(v, 0) + 1
                
                majority_label = max(vote_counts, key=vote_counts.get)
                labels.append(majority_label)
                
                # Confidence = agreement fraction
                agreement = vote_counts[majority_label] / len(votes)
                confidences.append(agreement)
        
        df["program_label"] = labels
        if return_confidence:
            df["program_confidence"] = confidences
        
        # Log coverage
        coverage = (df["program_label"].notna()).sum() / len(df)
        logger.info(f"Program coverage: {coverage:.2%}")
        
        return df
    
    def _compile_program(self, code: str) -> Callable[[str], str | None]:
        """Compile program code into executable function.
        
        Parameters:
            code: Python source code defining label_fn
            
        Returns:
            Compiled label_fn function
        """
        # Parse and validate AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax: {e}")
        
        # Execute in isolated namespace
        namespace: dict[str, Any] = {}
        exec(compile(tree, '<string>', 'exec'), namespace)
        
        if "label_fn" not in namespace:
            raise ValueError("Program must define 'label_fn' function")
        
        return namespace["label_fn"]
    
    def save_programs(self, path: str | Path) -> None:
        """Save validated programs and stats to JSON.
        
        Parameters:
            path: Output JSON path
        """
        output = {
            "programs": self.programs,
            "stats": self.program_stats,
            "config": {
                "precision_threshold": self.precision_threshold,
                "coverage_threshold": self.coverage_threshold,
            },
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved {len(self.programs)} programs to {path}")
    
    @classmethod
    def load_programs(cls, path: str | Path) -> ProgramLabeler:
        """Load programs from JSON.
        
        Parameters:
            path: Input JSON path
            
        Returns:
            ProgramLabeler with loaded programs
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        config = data.get("config", {})
        labeler = cls(
            precision_threshold=config.get("precision_threshold", 0.8),
            coverage_threshold=config.get("coverage_threshold", 0.1),
        )
        
        labeler.programs = data["programs"]
        labeler.program_stats = data["stats"]
        
        logger.info(f"Loaded {len(labeler.programs)} programs from {path}")
        
        return labeler
