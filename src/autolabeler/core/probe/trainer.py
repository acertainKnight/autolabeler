"""Probe model trainer: fine-tune a HuggingFace classifier on distillation data.

Wraps the HuggingFace Trainer for a simple, reproducible fine-tuning loop.
Designed for fast local iteration -- train in minutes, get metrics immediately,
fix the training data, and repeat before running the full cloud pipeline.

Key design decisions:
    - Training weights: per-sample loss scaling from the distillation JSONL so
      high-quality (ACCEPT) and human-verified labels dominate over SOFT/QUARANTINE
      tier samples in the loss function.
    - Stratified split: label-proportional train/val so every class appears in
      both sets at roughly the same ratio (important for imbalanced label sets
      like hawk/dove where neutrals dominate).
    - EvaluationService integration: reuses the project's standard metrics
      (accuracy, ordinal F1, Cohen's kappa, confusion matrix) so probe results
      are directly comparable to full cloud model evaluations.

Dependencies (optional install group):
    pip install 'autolabeler[probe]'

    This pulls in transformers, torch, accelerate, and datasets. These are not
    required for the rest of the autolabeler pipeline and are kept optional to
    avoid bloating the default install.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from .config import ProbeConfig


class _WeightedTrainer:
    """Factory for a Trainer subclass that applies per-sample loss weights.

    The distillation export assigns a training_weight to each sample based on
    confidence tier (ACCEPT=1.0, SOFT=0.7, QUARANTINE=0.3). This class
    creates a Trainer subclass that uses those weights to scale the per-sample
    cross-entropy loss so high-quality labels influence the model more.

    Defined at module level (not inside ProbeTrainer) because the HuggingFace
    Trainer pickles the class for multi-process DataLoader workers.
    """

    @staticmethod
    def make(base_trainer_cls: type, sample_weights: np.ndarray) -> type:
        """Dynamically create a subclass with a custom compute_loss.

        Args:
            base_trainer_cls: HuggingFace Trainer class to subclass.
            sample_weights: Per-sample weight array aligned to the full dataset.

        Returns:
            A Trainer subclass that applies sample_weights during loss computation.
        """

        class _Trainer(base_trainer_cls):  # type: ignore[valid-type]
            _weights = sample_weights

            def compute_loss(
                self,
                model: Any,
                inputs: dict[str, Any],
                return_outputs: bool = False,
                **kwargs: Any,
            ) -> Any:
                import torch

                labels = inputs.pop('labels')
                outputs = model(**inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                # Standard cross-entropy, unreduced
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(logits, labels)

                # Sample weights: look up by the indices stored on the batch
                # We attach indices via the dataset __getitem__ using the key '_idx'.
                if '_idx' in inputs:
                    idx = inputs['_idx'].cpu().numpy()
                    weights = torch.tensor(
                        self._weights[idx], dtype=loss.dtype, device=loss.device
                    )
                    loss = (loss * weights).mean()
                else:
                    loss = loss.mean()

                return (loss, outputs) if return_outputs else loss

        return _Trainer


class _IndexedDataset:
    """Wrapper that appends a ``_idx`` field to every batch item.

    The ``_WeightedTrainer`` needs each sample's position in the training set
    to look up the corresponding weight. HuggingFace datasets don't expose the
    integer index by default, so this wrapper adds it as ``_idx``.

    Args:
        hf_dataset: A HuggingFace datasets.Dataset instance.
    """

    def __init__(self, hf_dataset: Any) -> None:
        self._ds = hf_dataset

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = dict(self._ds[idx])
        item['_idx'] = idx
        return item


class ProbeTrainer:
    """Fine-tune a HuggingFace classification model on distillation data.

    Reads the JSONL produced by DistillationExporter, performs a stratified
    train/val split, fine-tunes the model with optional per-sample weights,
    and evaluates using the project's EvaluationService.

    Args:
        probe_config: Hyperparameters and runtime settings.
        dataset_config: DatasetConfig for label vocabulary and dataset name.

    Example:
        >>> from autolabeler.core.probe import ProbeTrainer, ProbeConfig
        >>> from autolabeler.core.dataset_config import DatasetConfig
        >>> cfg = ProbeConfig(model_name="roberta-base", epochs=3)
        >>> ds_cfg = DatasetConfig.from_yaml("configs/fed_headlines.yaml")
        >>> trainer = ProbeTrainer(cfg, ds_cfg)
        >>> results = trainer.train("outputs/fed_headlines/distillation.jsonl")
        >>> print(results["metrics"]["accuracy"])

        >>> # Evaluate a previously trained model on new data
        >>> metrics = trainer.evaluate_saved(
        ...     "outputs/fed_headlines/probe",
        ...     "outputs/fed_headlines/distillation_v2.jsonl",
        ... )
    """

    def __init__(self, probe_config: ProbeConfig, dataset_config: Any) -> None:
        """Initialise ProbeTrainer.

        Args:
            probe_config: ProbeConfig instance.
            dataset_config: DatasetConfig instance for label vocab + dataset name.
        """
        self.probe_cfg = probe_config
        self.ds_cfg = dataset_config
        self._label_list: list[str] = [str(lbl) for lbl in dataset_config.labels]
        self._label2id: dict[str, int] = {lbl: i for i, lbl in enumerate(self._label_list)}
        self._id2label: dict[int, str] = {i: lbl for lbl, i in self._label2id.items()}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_distillation_data(self, jsonl_path: str | Path) -> pd.DataFrame:
        """Load distillation JSONL into a DataFrame.

        Args:
            jsonl_path: Path to JSONL file from DistillationExporter.

        Returns:
            DataFrame with columns: text, hard_label, training_weight, source, tier.

        Raises:
            FileNotFoundError: If the JSONL file does not exist.

        Example:
            >>> df = trainer.load_distillation_data("outputs/fed_headlines/distillation.jsonl")
            >>> df.columns.tolist()
            ['text', 'hard_label', 'training_weight', 'source', 'tier']
        """
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            raise FileNotFoundError(f'Distillation data not found: {jsonl_path}')

        records = []
        with open(jsonl_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        df = pd.DataFrame(records)
        logger.info(f'Loaded {len(df)} records from {jsonl_path}')

        # Keep only rows whose label is in the config vocabulary
        before = len(df)
        df['hard_label'] = df['hard_label'].astype(str)
        df = df[df['hard_label'].isin(self._label_list)].reset_index(drop=True)
        if len(df) < before:
            logger.warning(
                f'Dropped {before - len(df)} records with labels not in config vocab '
                f'{self._label_list}'
            )

        return df

    def stratified_split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Stratified train/val split preserving label proportions.

        Args:
            df: Full labeled DataFrame from load_distillation_data().

        Returns:
            Tuple of (train_df, val_df).

        Example:
            >>> train_df, val_df = trainer.stratified_split(df)
            >>> len(train_df) / len(df)
            0.8
        """
        from sklearn.model_selection import train_test_split

        train_df, val_df = train_test_split(
            df,
            test_size=self.probe_cfg.val_split,
            stratify=df['hard_label'],
            random_state=self.probe_cfg.seed,
        )
        logger.info(
            f'Split: {len(train_df)} train / {len(val_df)} val '
            f'(val_split={self.probe_cfg.val_split})'
        )
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def _build_hf_dataset(
        self,
        df: pd.DataFrame,
        tokenizer: Any,
    ) -> Any:
        """Tokenize a DataFrame and return a HuggingFace Dataset.

        Args:
            df: DataFrame with 'text' and 'hard_label' columns.
            tokenizer: HuggingFace tokenizer instance.

        Returns:
            HuggingFace datasets.Dataset with 'input_ids', 'attention_mask',
            'labels' columns.
        """
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise ImportError(
                'datasets package is required. Install with: pip install datasets'
            ) from exc

        label_ids = df['hard_label'].map(self._label2id).tolist()

        def _tokenize(batch: dict[str, Any]) -> dict[str, Any]:
            return tokenizer(
                batch['text'],
                padding='max_length',
                truncation=True,
                max_length=self.probe_cfg.max_length,
            )

        hf_ds = Dataset.from_dict({'text': df['text'].tolist(), 'labels': label_ids})
        hf_ds = hf_ds.map(_tokenize, batched=True)
        hf_ds = hf_ds.remove_columns(['text'])
        hf_ds.set_format('torch')
        return hf_ds

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        training_data: str | Path,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Fine-tune the probe model and return evaluation metrics.

        Args:
            training_data: Path to distillation JSONL from DistillationExporter.
            output_dir: Directory to save model + tokenizer. Defaults to
                ProbeConfig.resolve_output_dir(dataset_name).

        Returns:
            Dict with keys:
                - metrics: accuracy, macro_f1, per-class metrics, confusion_matrix
                - val_predictions: DataFrame with text, true_label, pred_label
                - model_path: Path where the model was saved
                - train_size / val_size: dataset sizes

        Example:
            >>> results = trainer.train("outputs/fed_headlines/distillation.jsonl")
            >>> results["metrics"]["val_accuracy"]
            0.82
        """
        _check_transformers_available()

        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )

        dataset_name = getattr(self.ds_cfg, 'name', 'dataset')
        resolved_output_dir = Path(output_dir or self.probe_cfg.resolve_output_dir(dataset_name))
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        df = self.load_distillation_data(training_data)
        train_df, val_df = self.stratified_split(df)

        # Load tokenizer + model
        logger.info(f'Loading {self.probe_cfg.model_name} tokenizer and model...')
        tokenizer = AutoTokenizer.from_pretrained(self.probe_cfg.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.probe_cfg.model_name,
            num_labels=len(self._label_list),
            id2label=self._id2label,
            label2id=self._label2id,
            ignore_mismatched_sizes=True,
        )

        # Build HF datasets
        train_ds = self._build_hf_dataset(train_df, tokenizer)
        val_ds = self._build_hf_dataset(val_df, tokenizer)

        # Per-sample weights (train only)
        trainer_cls = Trainer
        if self.probe_cfg.use_training_weights and 'training_weight' in train_df.columns:
            weights = train_df['training_weight'].fillna(1.0).to_numpy(dtype=np.float32)
            trainer_cls = _WeightedTrainer.make(Trainer, weights)
            train_ds = _IndexedDataset(train_ds)
            logger.info('Using per-sample training weights from distillation JSONL')

        # Determine steps
        steps_per_epoch = max(1, len(train_df) // self.probe_cfg.batch_size)
        total_steps = steps_per_epoch * self.probe_cfg.epochs

        # Check if CUDA is available for fp16
        import torch
        use_fp16 = self.probe_cfg.fp16 and torch.cuda.is_available()

        training_args = TrainingArguments(
            output_dir=str(resolved_output_dir),
            num_train_epochs=self.probe_cfg.epochs,
            per_device_train_batch_size=self.probe_cfg.batch_size,
            per_device_eval_batch_size=self.probe_cfg.batch_size * 2,
            learning_rate=self.probe_cfg.learning_rate,
            weight_decay=self.probe_cfg.weight_decay,
            warmup_ratio=self.probe_cfg.warmup_ratio,
            fp16=use_fp16,
            seed=self.probe_cfg.seed,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=self.probe_cfg.load_best_model_at_end,
            metric_for_best_model=self.probe_cfg.metric_for_best_model,
            save_total_limit=self.probe_cfg.save_total_limit,
            logging_steps=min(self.probe_cfg.logging_steps, max(1, total_steps // 10)),
            report_to='none',
            dataloader_num_workers=0,
        )

        def _compute_metrics(eval_pred: Any) -> dict[str, float]:
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            accuracy = float((preds == labels).mean())
            return {'accuracy': accuracy}

        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            compute_metrics=_compute_metrics,
        )

        logger.info(
            f'Training {self.probe_cfg.model_name} for {self.probe_cfg.epochs} epochs '
            f'on {len(train_df)} samples (val: {len(val_df)})...'
        )
        trainer.train()

        # Save model + tokenizer
        trainer.save_model(str(resolved_output_dir))
        tokenizer.save_pretrained(str(resolved_output_dir))
        logger.info(f'Model saved to {resolved_output_dir}')

        # Evaluation
        logger.info('Running final evaluation...')
        predictions = trainer.predict(val_ds)
        pred_ids = np.argmax(predictions.predictions, axis=-1)
        true_ids = predictions.label_ids
        pred_labels = [self._id2label[int(p)] for p in pred_ids]
        true_labels = [self._id2label[int(t)] for t in true_ids]

        metrics = self._evaluate(true_labels, pred_labels, dataset_name)

        val_predictions = val_df[['text', 'hard_label']].copy()
        val_predictions = val_predictions.rename(columns={'hard_label': 'true_label'})
        val_predictions['pred_label'] = pred_labels

        # Save predictions
        preds_path = resolved_output_dir / 'val_predictions.csv'
        val_predictions.to_csv(preds_path, index=False)
        logger.info(f'Validation predictions saved to {preds_path}')

        # Save metrics
        metrics_path = resolved_output_dir / 'eval_metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f'Evaluation metrics saved to {metrics_path}')

        return {
            'metrics': metrics,
            'val_predictions': val_predictions,
            'model_path': str(resolved_output_dir),
            'train_size': len(train_df),
            'val_size': len(val_df),
        }

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        true_labels: list[str],
        pred_labels: list[str],
        dataset_name: str,
    ) -> dict[str, Any]:
        """Compute metrics using EvaluationService.

        Args:
            true_labels: Ground-truth string labels.
            pred_labels: Predicted string labels.
            dataset_name: Used to select ordinal metrics (hawk_dove task).

        Returns:
            Dict with accuracy, macro/weighted F1, per-class metrics,
            confusion matrix, and ordinal metrics (if applicable).
        """
        try:
            from autolabeler.core.evaluation.evaluation_service import EvaluationService

            eval_df = pd.DataFrame({
                'true_label': true_labels,
                'pred_label': pred_labels,
            })
            service = EvaluationService(dataset_name)
            results = service.evaluate(
                df=eval_df,
                true_label_column='true_label',
                pred_label_column='pred_label',
                use_comprehensive_metrics=True,
                task_name='hawk_dove' if 'fed' in dataset_name.lower() else dataset_name,
            )
            metrics = results.get('metrics', {})

        except Exception as exc:
            logger.warning(f'EvaluationService failed, falling back to sklearn: {exc}')
            metrics = self._sklearn_metrics(true_labels, pred_labels)

        # Always log a clean summary
        acc = metrics.get('accuracy', 0.0)
        f1_macro = metrics.get('f1_macro', metrics.get('macro_f1', 0.0))
        f1_weighted = metrics.get('f1_weighted', metrics.get('weighted_f1', 0.0))
        logger.info(
            f'Probe evaluation -- accuracy: {acc:.4f} | '
            f'macro F1: {f1_macro:.4f} | '
            f'weighted F1: {f1_weighted:.4f}'
        )

        return metrics

    def _sklearn_metrics(
        self,
        true_labels: list[str],
        pred_labels: list[str],
    ) -> dict[str, Any]:
        """Compute basic metrics with sklearn as fallback.

        Args:
            true_labels: Ground-truth labels.
            pred_labels: Predicted labels.

        Returns:
            Dict with accuracy, macro_f1, weighted_f1, confusion_matrix.
        """
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
        )

        accuracy = float(accuracy_score(true_labels, pred_labels))
        f1_macro = float(f1_score(true_labels, pred_labels, average='macro', zero_division=0))
        f1_weighted = float(f1_score(true_labels, pred_labels, average='weighted', zero_division=0))
        labels = sorted(set(true_labels) | set(pred_labels))
        cm = confusion_matrix(true_labels, pred_labels, labels=labels).tolist()

        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm,
            'labels': labels,
        }

    # ------------------------------------------------------------------
    # Convenience: evaluate an already-saved model
    # ------------------------------------------------------------------

    def evaluate_saved(
        self,
        model_dir: str | Path,
        eval_data: str | Path,
    ) -> dict[str, Any]:
        """Evaluate a previously saved probe model on new data.

        Args:
            model_dir: Directory with saved model + tokenizer (from train()).
            eval_data: Path to distillation JSONL to evaluate on.

        Returns:
            Same metrics dict as train().

        Example:
            >>> results = trainer.evaluate_saved(
            ...     "outputs/fed_headlines/probe",
            ...     "outputs/fed_headlines/distillation.jsonl",
            ... )
        """
        _check_transformers_available()

        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

        model_dir = Path(model_dir)
        logger.info(f'Loading model from {model_dir}')
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

        df = self.load_distillation_data(eval_data)
        eval_ds = self._build_hf_dataset(df, tokenizer)

        # Minimal training args just for predict()
        tmp_args = TrainingArguments(
            output_dir=str(model_dir / 'tmp_eval'),
            per_device_eval_batch_size=self.probe_cfg.batch_size * 2,
            report_to='none',
            fp16=False,
        )
        evaluator = Trainer(model=model, args=tmp_args, tokenizer=tokenizer)

        predictions = evaluator.predict(eval_ds)
        pred_ids = np.argmax(predictions.predictions, axis=-1)
        true_ids = predictions.label_ids
        pred_labels = [self._id2label[int(p)] for p in pred_ids]
        true_labels = [self._id2label[int(t)] for t in true_ids]

        dataset_name = getattr(self.ds_cfg, 'name', 'dataset')
        return self._evaluate(true_labels, pred_labels, dataset_name)


def _check_transformers_available() -> None:
    """Raise a clear ImportError if transformers is not installed.

    Raises:
        ImportError: If transformers is not available.
    """
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            'transformers is required for the probe model. '
            "Install with: pip install 'autolabeler[probe]'"
        ) from exc
