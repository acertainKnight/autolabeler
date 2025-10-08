"""
Direct Preference Optimization (DPO) service for task-specific LLM alignment.

This module provides functionality for aligning language models to specific
annotation tasks using human preferences and corrections.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field


class DPOServiceConfig(BaseModel):
    """Configuration for DPO alignment service."""

    base_model: str = Field(description="Base model name for fine-tuning")
    output_dir: str = Field(description="Directory for saving aligned model")
    num_epochs: int = Field(3, gt=0, description="Number of training epochs")
    learning_rate: float = Field(5e-5, gt=0, description="Learning rate")
    batch_size: int = Field(4, gt=0, description="Training batch size")
    gradient_accumulation_steps: int = Field(
        4, gt=0, description="Gradient accumulation steps"
    )
    beta: float = Field(
        0.1, gt=0, description="DPO regularization parameter"
    )
    max_length: int = Field(512, gt=0, description="Maximum sequence length")
    warmup_steps: int = Field(100, gt=0, description="Warmup steps")
    save_strategy: str = Field("epoch", description="Model save strategy")
    logging_steps: int = Field(10, gt=0, description="Logging frequency")


class DPOAlignmentService:
    """
    Direct Preference Optimization for task-specific LLM alignment.

    This service collects preference pairs from human corrections and uses
    them to fine-tune models for better task-specific performance.

    Example:
        >>> config = DPOServiceConfig(
        ...     base_model="meta-llama/Llama-3.1-8B-Instruct",
        ...     output_dir="models/aligned"
        ... )
        >>> service = DPOAlignmentService(config)
        >>> preferences = service.collect_preferences(
        ...     df, "text", "human_label", "model_label"
        ... )
        >>> service.train_dpo(preferences)
    """

    def __init__(self, config: DPOServiceConfig):
        """
        Initialize DPO alignment service.

        Args:
            config: DPO service configuration.
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self._model_loaded = False

    def initialize_model(self, model_name: str | None = None) -> None:
        """
        Initialize base model for fine-tuning.

        Args:
            model_name: Model name (uses config if not provided).
        """
        model_name = model_name or self.config.base_model

        try:
            # Try to import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading model: {model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

            # Set padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self._model_loaded = True
            logger.info("Model loaded successfully")

        except ImportError:
            logger.error(
                "transformers library not installed. "
                "Install with: pip install transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def collect_preferences(
        self,
        dataset: pd.DataFrame,
        text_column: str,
        human_label_column: str,
        model_label_column: str,
        task_description: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Collect preference pairs from human corrections.

        Creates training data where human corrections are "chosen" and
        model predictions are "rejected" when they disagree.

        Args:
            dataset: DataFrame with annotations.
            text_column: Column containing text.
            human_label_column: Column with human labels.
            model_label_column: Column with model labels.
            task_description: Optional task description for prompts.

        Returns:
            List of preference pair dictionaries.
        """
        preferences = []

        for _, row in dataset.iterrows():
            text = row[text_column]
            human_label = row[human_label_column]
            model_label = row[model_label_column]

            # Only create pairs where human corrected the model
            if human_label != model_label:
                prompt = self._create_labeling_prompt(text, task_description)

                preferences.append(
                    {
                        "prompt": prompt,
                        "chosen": self._format_response(
                            human_label, "Human-corrected label"
                        ),
                        "rejected": self._format_response(
                            model_label, "Model prediction"
                        ),
                    }
                )

        logger.info(f"Collected {len(preferences)} preference pairs")
        return preferences

    def train_dpo(
        self,
        preference_data: list[dict[str, Any]],
        output_dir: str | None = None,
        num_epochs: int | None = None,
        learning_rate: float | None = None,
    ) -> dict[str, Any]:
        """
        Train model using Direct Preference Optimization.

        Args:
            preference_data: List of preference pair dictionaries.
            output_dir: Output directory (uses config if not provided).
            num_epochs: Number of epochs (uses config if not provided).
            learning_rate: Learning rate (uses config if not provided).

        Returns:
            Training results dictionary.
        """
        if not self._model_loaded:
            self.initialize_model()

        output_dir = output_dir or self.config.output_dir
        num_epochs = num_epochs or self.config.num_epochs
        learning_rate = learning_rate or self.config.learning_rate

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        try:
            # Try to use TRL library for DPO
            from trl import DPOConfig, DPOTrainer

            # Configure DPO training
            training_args = DPOConfig(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=learning_rate,
                save_strategy=self.config.save_strategy,
                logging_steps=self.config.logging_steps,
                warmup_steps=self.config.warmup_steps,
                beta=self.config.beta,
                max_length=self.config.max_length,
                remove_unused_columns=False,
            )

            # Prepare dataset
            train_dataset = self._prepare_dataset(preference_data)

            # Initialize DPO trainer
            self.trainer = DPOTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=self.tokenizer,
            )

            logger.info("Starting DPO training...")
            training_output = self.trainer.train()

            # Save aligned model
            self.trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)

            logger.info(f"Model saved to {output_dir}")

            return {
                "output_dir": output_dir,
                "training_loss": training_output.training_loss,
                "num_epochs": num_epochs,
                "num_preferences": len(preference_data),
            }

        except ImportError:
            logger.error(
                "trl library not installed. Install with: pip install trl"
            )
            raise
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def evaluate_alignment(
        self,
        test_dataset: pd.DataFrame,
        text_column: str,
        label_column: str,
    ) -> dict[str, Any]:
        """
        Evaluate aligned model performance.

        Args:
            test_dataset: Test DataFrame.
            text_column: Column containing text.
            label_column: Column with true labels.

        Returns:
            Evaluation metrics dictionary.
        """
        if not self._model_loaded:
            raise ValueError("Model not loaded. Call initialize_model() first.")

        predictions = []

        for _, row in test_dataset.iterrows():
            prompt = self._create_labeling_prompt(row[text_column])
            prediction = self._generate_prediction(prompt)
            predictions.append(prediction)

        # Calculate metrics
        try:
            from sklearn.metrics import accuracy_score, f1_score

            accuracy = accuracy_score(test_dataset[label_column], predictions)
            f1 = f1_score(
                test_dataset[label_column],
                predictions,
                average="weighted",
                zero_division=0,
            )

            results = {
                "accuracy": float(accuracy),
                "f1_score": float(f1),
                "num_test_samples": len(test_dataset),
            }

            logger.info(f"Evaluation results: {results}")
            return results

        except ImportError:
            logger.error(
                "scikit-learn not installed. Install with: pip install scikit-learn"
            )
            raise

    def _create_labeling_prompt(
        self, text: str, task_description: str | None = None
    ) -> str:
        """
        Create labeling prompt for text.

        Args:
            text: Text to label.
            task_description: Optional task description.

        Returns:
            Formatted prompt string.
        """
        if task_description:
            return f"""Task: {task_description}

Text: {text}

Label:"""
        else:
            return f"""Provide a label for the following text:

{text}

Label:"""

    def _format_response(self, label: str, source: str) -> str:
        """
        Format response for preference pair.

        Args:
            label: Label value.
            source: Source description.

        Returns:
            Formatted response string.
        """
        return f"{label}"

    def _generate_prediction(self, prompt: str) -> str:
        """
        Generate prediction from model.

        Args:
            prompt: Input prompt.

        Returns:
            Predicted label.
        """
        if not self._model_loaded:
            raise ValueError("Model not loaded")

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=False,
            )
            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract label from response
            # Simple extraction - in practice would need more robust parsing
            prediction = prediction.replace(prompt, "").strip()

            return prediction

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return "unknown"

    def _prepare_dataset(self, preference_data: list[dict[str, Any]]) -> Any:
        """
        Prepare dataset for DPO training.

        Args:
            preference_data: List of preference pairs.

        Returns:
            Prepared dataset object.
        """
        from datasets import Dataset

        # Convert to Dataset format
        dataset = Dataset.from_list(preference_data)

        return dataset

    def load_aligned_model(self, model_path: str) -> None:
        """
        Load a previously aligned model.

        Args:
            model_path: Path to aligned model.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading aligned model from: {model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self._model_loaded = True
            logger.info("Aligned model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load aligned model: {e}")
            raise

    def save_preferences(
        self, preferences: list[dict[str, Any]], output_path: str
    ) -> None:
        """
        Save preference pairs to file.

        Args:
            preferences: List of preference pairs.
            output_path: Output file path (JSONL format).
        """
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for pref in preferences:
                f.write(json.dumps(pref) + "\n")

        logger.info(f"Saved {len(preferences)} preferences to {output_path}")

    def load_preferences(self, input_path: str) -> list[dict[str, Any]]:
        """
        Load preference pairs from file.

        Args:
            input_path: Input file path (JSONL format).

        Returns:
            List of preference pairs.
        """
        import json

        preferences = []

        with open(input_path, "r") as f:
            for line in f:
                preferences.append(json.loads(line.strip()))

        logger.info(f"Loaded {len(preferences)} preferences from {input_path}")
        return preferences
