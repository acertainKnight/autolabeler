"""Export labeled data for model distillation with confidence-based training weights.

This module implements the distillation export pipeline with:
- Confidence-weighted training weights based on tier and verification status
- Soft label distributions for uncertain cases
- Human label mixing and oversampling
- Source tracking for differential loss weighting

Evidence base:
- SiDyP 2025: ~7% improvement training on properly weighted LLM labels
- LiLAW 2025: dynamic sample weighting with 3 learnable parameters
- "A Little Human Data Goes A Long Way" (arXiv 2410.13098): 125 human examples
  dramatically improve distilled model quality
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


class DistillationExporter:
    """Exports labeled data optimized for model distillation.
    
    Combines LLM-labeled data with optional human labels, applying appropriate
    training weights based on confidence, agreement, and verification status.
    
    Attributes:
        weight_config: Dict mapping tier names to training weights
        
    Example:
        >>> exporter = DistillationExporter()
        >>> exporter.export(
        ...     llm_labeled_csv="outputs/fed_headlines/labeled.csv",
        ...     output_path="outputs/fed_headlines/distillation.jsonl",
        ...     human_labeled_csv="datasets/human_labeled.csv",
        ...     human_oversample=3.0
        ... )
    """
    
    def __init__(
        self,
        weight_config: dict[str, float] | None = None,
    ):
        """Initialize distillation exporter.
        
        Parameters:
            weight_config: Optional custom weight mapping (tier -> weight)
        """
        # Default weight formula based on tier and verification
        self.weight_config = weight_config or {
            "ACCEPT_verified": 1.0,
            "ACCEPT_unverified": 0.9,
            "ACCEPT-M": 0.7,
            "SOFT": 0.5,
            "QUARANTINE": 0.0,
            "human": 1.2,  # Oversampled for importance
        }
    
    def export(
        self,
        llm_labeled_csv: str | Path,
        output_path: str | Path,
        human_labeled_csv: str | Path | None = None,
        human_text_column: str = "text",
        human_label_column: str = "label",
        human_oversample: float = 3.0,
        exclude_quarantine: bool = True,
    ) -> dict[str, Any]:
        """Export LLM-labeled data with optional human label mixing.
        
        Parameters:
            llm_labeled_csv: Path to LLM-labeled data (output from run_labeling.py)
            output_path: Path to save distillation-ready JSONL
            human_labeled_csv: Optional path to human-labeled data
            human_text_column: Column name for text in human data
            human_label_column: Column name for label in human data
            human_oversample: Factor to oversample human labels (default 3.0)
            exclude_quarantine: Whether to exclude QUARANTINE tier (default True)
            
        Returns:
            Dict with export statistics
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load LLM-labeled data
        logger.info(f"Loading LLM-labeled data from {llm_labeled_csv}")
        llm_df = pd.read_csv(llm_labeled_csv)
        
        # Convert to distillation format
        records = []
        for _, row in llm_df.iterrows():
            # Skip quarantine if configured
            if exclude_quarantine and row.get("tier") == "QUARANTINE":
                continue
            
            # Determine training weight
            tier = row.get("tier", "ACCEPT")
            verified = row.get("verified", False) if "verified" in row else False
            
            if tier == "ACCEPT" and verified:
                weight_key = "ACCEPT_verified"
            elif tier == "ACCEPT" and not verified:
                weight_key = "ACCEPT_unverified"
            else:
                weight_key = tier
            
            training_weight = self.weight_config.get(weight_key, 0.8)
            
            # Parse soft label if present
            soft_label = None
            if pd.notna(row.get("soft_label")):
                try:
                    soft_label = json.loads(row["soft_label"])
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # Build record
            record = {
                "text": row.get("text") or row.get("headline") or "",
                "hard_label": str(row["label"]),
                "soft_label": soft_label or {str(row["label"]): 1.0},
                "training_weight": training_weight,
                "source": "llm",
                "tier": tier,
                "verified": verified,
                "confidence": row.get("confidence", 0.7) if "confidence" in row else None,
                "agreement": row.get("agreement", "unknown"),
            }
            records.append(record)
        
        logger.info(f"Processed {len(records)} LLM-labeled records")
        
        # Mix in human labels if provided
        if human_labeled_csv:
            human_records = self._mix_human_labels(
                human_labeled_csv,
                human_text_column,
                human_label_column,
                human_oversample,
            )
            records.extend(human_records)
            logger.info(f"Added {len(human_records)} human-labeled records (oversampled {human_oversample}x)")
        
        # Write to JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
        
        logger.info(f"Exported {len(records)} records to {output_path}")
        
        # Compute statistics
        stats = self._compute_statistics(records)
        stats_path = output_path.with_suffix('.stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dumps(stats, f, indent=2)
        
        logger.info(f"Export statistics saved to {stats_path}")
        
        return stats
    
    def _mix_human_labels(
        self,
        human_csv: str | Path,
        text_column: str,
        label_column: str,
        oversample: float,
    ) -> list[dict[str, Any]]:
        """Load and oversample human-labeled data.
        
        Parameters:
            human_csv: Path to human-labeled CSV
            text_column: Column containing text
            label_column: Column containing labels
            oversample: Factor to repeat human examples
            
        Returns:
            List of human-labeled records (oversampled)
        """
        df = pd.read_csv(human_csv)
        
        if text_column not in df.columns:
            raise ValueError(
                f"Text column '{text_column}' not found in human data. "
                f"Available: {df.columns.tolist()}"
            )
        if label_column not in df.columns:
            raise ValueError(
                f"Label column '{label_column}' not found in human data. "
                f"Available: {df.columns.tolist()}"
            )
        
        # Drop rows with missing values
        df = df.dropna(subset=[text_column, label_column])
        
        records = []
        for _, row in df.iterrows():
            record = {
                "text": row[text_column],
                "hard_label": str(row[label_column]),
                "soft_label": {str(row[label_column]): 1.0},
                "training_weight": self.weight_config["human"],
                "source": "human",
                "tier": "HUMAN",
                "verified": True,
                "confidence": 1.0,
                "agreement": "gold_standard",
            }
            
            # Oversample by repeating
            n_copies = max(1, int(oversample))
            for _ in range(n_copies):
                records.append(record.copy())
        
        return records
    
    def _compute_statistics(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute export statistics.
        
        Parameters:
            records: List of export records
            
        Returns:
            Dict with statistics
        """
        # Source distribution
        source_counts = {}
        for r in records:
            source = r["source"]
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Tier distribution
        tier_counts = {}
        for r in records:
            tier = r["tier"]
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        # Label distribution
        label_counts = {}
        for r in records:
            label = r["hard_label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Soft vs hard label count
        n_soft = sum(1 for r in records if len(r["soft_label"]) > 1)
        n_hard = len(records) - n_soft
        
        # Average training weight by source
        source_avg_weight = {}
        for source in source_counts:
            source_records = [r for r in records if r["source"] == source]
            avg_weight = sum(r["training_weight"] for r in source_records) / len(source_records)
            source_avg_weight[source] = avg_weight
        
        return {
            "total_records": len(records),
            "source_distribution": source_counts,
            "tier_distribution": tier_counts,
            "label_distribution": label_counts,
            "soft_label_count": n_soft,
            "hard_label_count": n_hard,
            "source_avg_weight": source_avg_weight,
            "weight_config": self.weight_config,
        }
