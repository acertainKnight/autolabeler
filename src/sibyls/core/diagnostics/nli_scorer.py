"""NLI-based coherence scoring for (text, label) pairs.

Frames each annotation as a Natural Language Inference problem:
    Premise: the text being labeled
    Hypothesis: a natural-language description of what the label means

This provides a model-independent verification signal that is orthogonal to
the jury system -- the NLI model has never seen the labeling prompts, so
correlated errors between jury and NLI are rare. Disagreements between the
two systems are the highest-value targets for human review.

Requires a hypotheses.yaml file mapping each label to a hypothesis string.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from .config import DiagnosticsConfig


class NLIScorer:
    """Score (text, label) coherence using a cross-encoder NLI model.

    Entailment scoring asks: given this text as the premise, does the label
    hypothesis logically follow? Low entailment = potential mismatch.

    Contrastive scoring computes entailment for ALL label hypotheses and flags
    samples where an alternative label has higher entailment than the assigned one.

    Args:
        config: DiagnosticsConfig with nli_model and threshold settings.

    Example:
        >>> scorer = NLIScorer(config)
        >>> hypotheses = scorer.load_hypotheses("prompts/fed_headlines/hypotheses.yaml")
        >>> results = scorer.run_all(texts, labels, hypotheses)
    """

    def __init__(self, config: DiagnosticsConfig) -> None:
        """Initialize NLI scorer. Model loading is deferred to first use.

        Args:
            config: DiagnosticsConfig instance.
        """
        self.config = config
        self._model = None

    def _load_model(self) -> None:
        """Lazily load the cross-encoder NLI model."""
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        max_len = self.config.nli_max_length
        logger.info(
            f'Loading NLI model: {self.config.nli_model}'
            f'{f" (max_length={max_len})" if max_len else ""}'
        )
        self._model = CrossEncoder(self.config.nli_model, max_length=max_len)

    def load_hypotheses(self, path: str | Path) -> dict[str, str]:
        """Load label-to-hypothesis mapping from YAML file.

        The YAML file must have a 'labels' key mapping label values to natural
        language hypothesis strings.

        Args:
            path: Path to hypotheses.yaml file.

        Returns:
            Dictionary mapping str(label) -> hypothesis string.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the file is missing required keys.

        Example:
            >>> hypotheses = scorer.load_hypotheses("prompts/fed_headlines/hypotheses.yaml")
            >>> hypotheses["-2"]
            'This text expresses a strongly dovish monetary policy stance'
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f'Hypotheses file not found: {path}. '
                'Create a hypotheses.yaml with a "labels" key mapping each label to a hypothesis.'
            )

        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if 'labels' not in data:
            raise ValueError(
                f'hypotheses.yaml at {path} must have a top-level "labels" key. '
                'Example: labels:\n  "0": "This text is neutral regarding..."'
            )

        hypotheses = {str(k): str(v) for k, v in data['labels'].items()}
        logger.info(f'Loaded {len(hypotheses)} label hypotheses from {path}')
        return hypotheses

    @staticmethod
    def _extract_entailment(raw_scores: np.ndarray) -> np.ndarray:
        """Extract entailment probabilities from raw NLI model output.

        Args:
            raw_scores: Model output, shape (n,) or (n, 3).

        Returns:
            1-D array of entailment probabilities, length n.
        """
        if raw_scores.ndim == 2 and raw_scores.shape[1] == 3:
            return raw_scores[:, 2].astype(float)
        return raw_scores.astype(float)

    def score_entailment(
        self,
        texts: list[str],
        labels: list[str],
        hypotheses: dict[str, str],
    ) -> pd.DataFrame:
        """Compute NLI entailment score for each (text, assigned_label) pair.

        Constructs premise=text, hypothesis=label_hypothesis and scores with the
        NLI cross-encoder. Samples with entailment below threshold are flagged.

        Args:
            texts: Input texts (premises).
            labels: Assigned label for each text.
            hypotheses: Mapping from label string to hypothesis string.

        Returns:
            DataFrame with columns:
                - index: sample position
                - label: assigned label
                - hypothesis: NLI hypothesis used
                - entailment_score: model's entailment probability (0-1)
                - is_flagged: True if entailment_score < threshold

        Example:
            >>> df = scorer.score_entailment(texts, labels, hypotheses)
            >>> df[df["is_flagged"]].head()
        """
        self._load_model()

        pairs = []
        valid_indices = []
        for i, (text, lbl) in enumerate(zip(texts, labels)):
            hyp = hypotheses.get(str(lbl))
            if hyp is None:
                logger.debug(f'No hypothesis for label {lbl} -- skipping sample {i}')
                continue
            pairs.append((text, hyp))
            valid_indices.append(i)

        if not pairs:
            logger.warning('No valid (text, hypothesis) pairs found for entailment scoring')
            return pd.DataFrame(columns=['index', 'label', 'hypothesis', 'entailment_score', 'is_flagged'])

        logger.info(f'Scoring entailment for {len(pairs)} samples...')
        raw_scores = self._model.predict(
            pairs, apply_softmax=True, batch_size=self.config.nli_batch_size,
        )
        entailment_scores = self._extract_entailment(raw_scores)
        threshold = self.config.nli_entailment_threshold

        records = []
        for local_i, global_i in enumerate(valid_indices):
            lbl = str(labels[global_i])
            score = entailment_scores[local_i]
            records.append({
                'index': global_i,
                'label': lbl,
                'hypothesis': hypotheses.get(lbl, ''),
                'entailment_score': score,
                'is_flagged': score < threshold,
            })

        return pd.DataFrame(records).sort_values('entailment_score')

    def score_contrastive(
        self,
        texts: list[str],
        labels: list[str],
        hypotheses: dict[str, str],
    ) -> pd.DataFrame:
        """Score entailment for ALL label hypotheses per sample (batched).

        Flags samples where an alternative label hypothesis has higher entailment
        than the assigned label. Ranked by margin (negative = likely wrong label).

        Margin = entailment(assigned) - max(entailment(others)).
        Negative margin indicates a likely labeling error.

        All (text, hypothesis) pairs are collected up-front and scored in a
        single batched ``predict()`` call for dramatically better throughput
        compared to per-sample inference.

        Args:
            texts: Input texts.
            labels: Assigned label for each text.
            hypotheses: Full mapping from all labels to hypothesis strings.

        Returns:
            DataFrame with columns:
                - index: sample position
                - label: assigned label
                - assigned_entailment: entailment score for assigned label
                - best_alternative_label: label with highest alternative entailment
                - best_alternative_entailment: that score
                - margin: assigned_entailment - best_alternative_entailment
                - is_contrastive_violation: True when margin < 0

        Example:
            >>> df = scorer.score_contrastive(texts, labels, hypotheses)
            >>> errors = df[df["is_contrastive_violation"]].sort_values("margin")
        """
        self._load_model()

        all_label_keys = list(hypotheses.keys())
        n_labels = len(all_label_keys)

        all_pairs: list[tuple[str, str]] = []
        valid_indices: list[int] = []

        for i, (text, assigned_lbl) in enumerate(zip(texts, labels)):
            if str(assigned_lbl) not in hypotheses:
                continue
            valid_indices.append(i)
            for lbl in all_label_keys:
                all_pairs.append((text, hypotheses[lbl]))

        if not all_pairs:
            logger.warning('No valid samples for contrastive scoring')
            return pd.DataFrame(columns=[
                'index', 'label', 'assigned_entailment',
                'best_alternative_label', 'best_alternative_entailment',
                'margin', 'is_contrastive_violation',
            ])

        logger.info(
            f'Contrastive scoring: {len(valid_indices)} samples × '
            f'{n_labels} labels = {len(all_pairs)} pairs'
        )
        raw_scores = self._model.predict(
            all_pairs, apply_softmax=True, batch_size=self.config.nli_batch_size,
        )
        entailment_scores = self._extract_entailment(raw_scores)

        # Reshape to (n_valid_samples, n_labels)
        score_matrix = entailment_scores.reshape(len(valid_indices), n_labels)

        records = []
        for row_i, global_i in enumerate(valid_indices):
            assigned_lbl = str(labels[global_i])
            assigned_col = all_label_keys.index(assigned_lbl)
            assigned_score = float(score_matrix[row_i, assigned_col])

            alt_scores = np.copy(score_matrix[row_i])
            alt_scores[assigned_col] = -np.inf
            best_alt_col = int(np.argmax(alt_scores))
            best_alt_score = float(score_matrix[row_i, best_alt_col])
            margin = assigned_score - best_alt_score

            scores_by_label = {
                lbl: float(score_matrix[row_i, j])
                for j, lbl in enumerate(all_label_keys)
            }

            records.append({
                'index': global_i,
                'label': assigned_lbl,
                'assigned_entailment': assigned_score,
                'best_alternative_label': all_label_keys[best_alt_col],
                'best_alternative_entailment': best_alt_score,
                'margin': margin,
                'is_contrastive_violation': margin < 0,
                'all_scores': scores_by_label,
            })

        return pd.DataFrame(records).sort_values('margin')

    def run_all(
        self,
        texts: list[str],
        labels: list[str],
        hypotheses: dict[str, str],
    ) -> dict[str, Any]:
        """Run both entailment and contrastive scoring.

        Args:
            texts: Input texts.
            labels: Assigned labels.
            hypotheses: Label-to-hypothesis mapping.

        Returns:
            Dictionary with keys:
                - entailment: DataFrame from score_entailment
                - contrastive: DataFrame from score_contrastive
                - summary: high-level counts

        Example:
            >>> results = scorer.run_all(texts, labels, hypotheses)
            >>> results["summary"]
        """
        logger.info(f'NLIScorer: analysing {len(texts)} samples')

        entailment_df = self.score_entailment(texts, labels, hypotheses)
        contrastive_df = self.score_contrastive(texts, labels, hypotheses)

        n_entailment_flagged = int(entailment_df['is_flagged'].sum()) if not entailment_df.empty else 0
        n_contrastive_violations = (
            int(contrastive_df['is_contrastive_violation'].sum())
            if not contrastive_df.empty
            else 0
        )

        logger.info(
            f'NLI analysis complete: '
            f'{n_entailment_flagged} entailment flags, '
            f'{n_contrastive_violations} contrastive violations'
        )

        return {
            'entailment': entailment_df,
            'contrastive': contrastive_df,
            'summary': {
                'n_entailment_flagged': n_entailment_flagged,
                'n_contrastive_violations': n_contrastive_violations,
            },
        }
